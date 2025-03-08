import datetime
import logging
import math
import time
import torch
import copy
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            # build the full training dataset
            full_train_set = build_dataset(dataset_opt)
            
            # if cross validation is enabled, split full_train_set
            if opt.get('cross_validation', {}).get('enable', False):
                from sklearn.model_selection import KFold
                n_folds = opt['cross_validation'].get('n_folds', 5)
                current_fold = opt['cross_validation'].get('current_fold', 0)
                logger.info(f'Performing {n_folds}-fold cross validation, current fold: {current_fold}')

                # Use the dataset length, not data_infos
                indices = list(range(len(full_train_set)))
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=opt['manual_seed'])
                splits = list(kf.split(indices))
                train_idx, cv_val_idx = splits[current_fold]

                from torch.utils.data import Subset
                train_set = Subset(full_train_set, train_idx)
                cv_val_set = Subset(full_train_set, cv_val_idx)
            else:
                train_set = full_train_set

            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            
            # If cross validation is enabled, use the validation split from train
            if opt.get('cross_validation', {}).get('enable', False):
                val_loader = build_dataloader(
                    cv_val_set,
                    dataset_opt,
                    num_gpu=opt['num_gpu'],
                    dist=opt['dist'],
                    sampler=None,
                    seed=opt['manual_seed'])
                val_loaders.append(val_loader)
                logger.info(f'Cross validation: {len(cv_val_set)} images for validation')
            else:
                # (Normal training: if not using cross validation, you may already have a separate val dataset)
                pass

            if not opt.get('cross_validation', {}).get('enable', False):
                # Log statistics for the full training dataset
                num_iter_per_epoch = math.ceil(
                    len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
                total_iters = int(opt['train']['total_iter'])
                total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
                logger.info('Training statistics:'
                            f'\n\tNumber of train images: {len(train_set)}'
                            f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                            f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                            f'\n\tWorld size (gpu number): {opt["world_size"]}'
                            f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                            f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
            else:
                # When using cross validation, you might want to set total_epochs and total_iters manually per fold.
                total_epochs = opt['train'].get('total_epochs', 100)  # or set this in your config
                total_iters = opt['train'].get('total_iter', 10000)
        elif phase.split('_')[0] == 'val':
            # In a cross validation run, you may ignore the separate 'val' dataset
            if not opt.get('cross_validation', {}).get('enable', False):
                val_set = build_dataset(dataset_opt)
                val_loader = build_dataloader(
                    val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
                logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
                val_loaders.append(val_loader)
            else:
                logger.info('Using cross validation split; ignoring external val dataset.')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
            
    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


# def create_train_val_dataloader(opt, logger):
#     # create train and val dataloaders
#     train_loader, val_loaders = None, []
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'train':
#             dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
#             train_set = build_dataset(dataset_opt)
#             train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
#             train_loader = build_dataloader(
#                 train_set,
#                 dataset_opt,
#                 num_gpu=opt['num_gpu'],
#                 dist=opt['dist'],
#                 sampler=train_sampler,
#                 seed=opt['manual_seed'])

#             num_iter_per_epoch = math.ceil(
#                 len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
#             total_iters = int(opt['train']['total_iter'])
#             total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
#             logger.info('Training statistics:'
#                         f'\n\tNumber of train images: {len(train_set)}'
#                         f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
#                         f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
#                         f'\n\tWorld size (gpu number): {opt["world_size"]}'
#                         f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
#                         f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
#         elif phase.split('_')[0] == 'val':
#             val_set = build_dataset(dataset_opt)
#             val_loader = build_dataloader(
#                 val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#             logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
#             val_loaders.append(val_loader)
#         else:
#             raise ValueError(f'Dataset phase {phase} is not recognized.')

#     return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state

def train_pipeline(root_path, override_opt=None):
    """
    If override_opt is not None, we skip parse_options and use override_opt as 'opt'.
    Otherwise, we parse the config from the .yml file as usual.
    """
    if override_opt is None:
        # parse options, set distributed setting, set random seed
        opt, args = parse_options(root_path, is_train=True)
    else:
        # use the override options directly
        opt = override_opt
        args = None  # not used if we already have 'opt'
    
    opt['root_path'] = root_path

    # --- The rest of your code remains the same ---
    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    copy_opt_file(args.opt if args else None, opt['path']['experiments_root'])  # if args is None, handle gracefully
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            iter_timer.record()
            if current_iter == 1:
                msg_logger.reset_start_time()

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


def do_training(opt):
    """
    This function does NOT call parse_options. Instead, it assumes 'opt' is already
    a fully populated dictionary. We'll do all training steps here.
    """
    # 1. Make sure we have a valid root_path and config path
    root_path = opt['root_path']
    opt_file_path = opt.get('opt_file', None)  # store the path from main

    torch.backends.cudnn.benchmark = True

    # 2. Resume state if necessary
    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(root_path, 'tb_logger', opt['name']))

    # 3. Copy the YAML config to experiments folder if we have a valid path
    if opt_file_path is not None and opt['path'].get('experiments_root') is not None:
        copy_opt_file(opt_file_path, opt['path']['experiments_root'])

    # 4. Initialize logger
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # 5. TensorBoard / WandB
    tb_logger = init_tb_loggers(opt)

    # 6. Create dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    # 7. Build model
    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # 8. Message logger
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 9. Prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # 10. Training loop
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            # update lr
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # train
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()

            if current_iter == 1:
                msg_logger.reset_start_time()

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    # end of epoch
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()



def main():  
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))  
    opt, args = parse_options(root_path, is_train=True)  
    opt['opt_file'] = args.opt  
    opt['root_path'] = root_path  

    if opt.get('cross_validation', {}).get('enable', False):  
        n_folds = opt['cross_validation'].get('n_folds', 5)  
        base_name = opt['name']  
        
        for fold in range(n_folds):  
            fold_opt = copy.deepcopy(opt)  
            fold_opt['cross_validation']['current_fold'] = fold  
            fold_opt['name'] = f"{base_name}_fold{fold}"  
            print(f"\n{'='*50}")  
            print(f"STARTING FOLD {fold+1}/{n_folds} - WILL RUN FOR {fold_opt['train']['total_iter']} ITERATIONS")  
            print(f"{'='*50}\n")  
            
            # Important: Clear or update resume path for each fold  
            if 'resume_state' in fold_opt['path']:  
                # Either clear it  
                del fold_opt['path']['resume_state']  
            
            print(f"\n========== Training on fold {fold+1} / {n_folds} ==========")  
            do_training(fold_opt)  # Call this only ONCE per fold  
            
            print(f"\n{'='*50}")  
            print(f"COMPLETED FOLD {fold+1}/{n_folds}")  
            print(f"{'='*50}\n")  
    else:  
        print("\n========== Training without cross-validation ==========")  
        do_training(opt)  

if __name__ == '__main__':
    main()


# if __name__ == '__main__':
#     import copy
#     root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     # Parse the options once
#     opt, args = parse_options(root_path, is_train=True)
    
#     # Check if cross-validation is enabled
#     if opt.get('cross_validation', {}).get('enable', False):
#         n_folds = opt['cross_validation'].get('n_folds', 5)
#         # Loop over each fold
#         for fold in range(n_folds):
#             # Create a deep copy of the original options for each fold run
#             fold_opt = copy.deepcopy(opt)
#             fold_opt['cross_validation']['current_fold'] = fold
#             # Optionally, update the experiment name so logs/models are saved separately:
#             fold_opt['name'] = f"{fold_opt['name']}_fold{fold}"
#             print(f"Training on fold {fold} / {n_folds}")
            
#             # You can call train_pipeline with fold_opt if you modify train_pipeline to accept an opt override.
#             # For example, modify train_pipeline to:
#             # def train_pipeline(root_path, override_opt=None):
#             #   if override_opt is None:
#             #       opt, args = parse_options(root_path, is_train=True)
#             #   else:
#             #       opt, args = override_opt, None
#             #   ... rest of the training pipeline ...
#             train_pipeline(root_path, override_opt=fold_opt)
            
#             # (Optional) Save or log the results for this fold before moving to the next one.
#     else:
#         # Normal single-run training
#         train_pipeline(root_path)




# def train_pipeline(root_path):
#     # parse options, set distributed setting, set ramdom seed
#     opt, args = parse_options(root_path, is_train=True)
#     opt['root_path'] = root_path

#     torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.deterministic = True

#     # load resume states if necessary
#     resume_state = load_resume_state(opt)
#     # mkdir for experiments and logger
#     if resume_state is None:
#         make_exp_dirs(opt)
#         if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
#             mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

#     # copy the yml file to the experiment root
#     copy_opt_file(args.opt, opt['path']['experiments_root'])

#     # WARNING: should not use get_root_logger in the above codes, including the called functions
#     # Otherwise the logger will not be properly initialized
#     log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))
#     # initialize wandb and tb loggers
#     tb_logger = init_tb_loggers(opt)

#     # create train and validation dataloaders
#     result = create_train_val_dataloader(opt, logger)
#     train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

#     # create model
#     model = build_model(opt)
#     if resume_state:  # resume training
#         model.resume_training(resume_state)  # handle optimizers and schedulers
#         logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
#         start_epoch = resume_state['epoch']
#         current_iter = resume_state['iter']
#     else:
#         start_epoch = 0
#         current_iter = 0

#     # create message logger (formatted outputs)
#     msg_logger = MessageLogger(opt, current_iter, tb_logger)

#     # dataloader prefetcher
#     prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
#     if prefetch_mode is None or prefetch_mode == 'cpu':
#         prefetcher = CPUPrefetcher(train_loader)
#     elif prefetch_mode == 'cuda':
#         prefetcher = CUDAPrefetcher(train_loader, opt)
#         logger.info(f'Use {prefetch_mode} prefetch dataloader')
#         if opt['datasets']['train'].get('pin_memory') is not True:
#             raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
#     else:
#         raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

#     # training
#     logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
#     data_timer, iter_timer = AvgTimer(), AvgTimer()
#     start_time = time.time()

#     for epoch in range(start_epoch, total_epochs + 1):
#         train_sampler.set_epoch(epoch)
#         prefetcher.reset()
#         train_data = prefetcher.next()

#         while train_data is not None:
#             data_timer.record()

#             current_iter += 1
#             if current_iter > total_iters:
#                 break
#             # update learning rate
#             model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
#             # training
#             model.feed_data(train_data)
#             model.optimize_parameters(current_iter)
#             iter_timer.record()
#             if current_iter == 1:
#                 # reset start time in msg_logger for more accurate eta_time
#                 # not work in resume mode
#                 msg_logger.reset_start_time()
#             # log
#             if current_iter % opt['logger']['print_freq'] == 0:
#                 log_vars = {'epoch': epoch, 'iter': current_iter}
#                 log_vars.update({'lrs': model.get_current_learning_rate()})
#                 log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
#                 log_vars.update(model.get_current_log())
#                 msg_logger(log_vars)

#             # save models and training states
#             if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
#                 logger.info('Saving models and training states.')
#                 model.save(epoch, current_iter)

#             # validation
#             if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
#                 if len(val_loaders) > 1:
#                     logger.warning('Multiple validation datasets are *only* supported by SRModel.')
#                 for val_loader in val_loaders:
#                     model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

#             data_timer.start()
#             iter_timer.start()
#             train_data = prefetcher.next()
#         # end of iter

#     # end of epoch

#     consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
#     logger.info(f'End of training. Time consumed: {consumed_time}')
#     logger.info('Save the latest model.')
#     model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
#     if opt.get('val') is not None:
#         for val_loader in val_loaders:
#             model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
#     if tb_logger:
#         tb_logger.close()


# if __name__ == '__main__':
#     root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     train_pipeline(root_path)

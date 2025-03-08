# PBVS2025 TISR Track 1 - 3rd Place


</details>

## Install
Create a conda enviroment:
````
ENV_NAME="seemore"
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
````
Run following script to install the dependencies:
````
bash install.sh
````

## Usage
Pre-trained checkpoints and visual results can be downloaded [here](https://drive.google.com/drive/folders/15jtvcS4jL_6QqEwaRodEN8FBrqVPrO2u?usp=share_link). Place the checkpoints in `checkpoints/`.

In `options` you can find the corresponding config files for reproducing our experiments.

##### **Testing**
For testing the pre-trained checkpoints please use following commands. Replace `[TEST OPT YML]` with the path to the corresponding option file.
`````
python basicsr/test.py -opt [TEST OPT YML]
`````

##### **Training**
For single-GPU training use the following commands. Replace `[TRAIN OPT YML]` with the path to the corresponding option file.
`````
torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt [TRAIN OPT YML] --launcher pytorch
`````

## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.


## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [SeemoRe](https://github.com/eduardzamfir/seemoredetails).



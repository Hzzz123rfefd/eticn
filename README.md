# ETICN: An Efficient Traffic Image Compression Network
Pytorch implementation of the paper "ETICN: An Efficient Traffic Image Compression Network". 
This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI). We kept some scripts, and removed other components. The major changes are provided in `compressai/models`. For the official code release, see the [CompressAI](https://github.com/InterDigitalInc/CompressAI).

## About
This paper defines the eticn model for learned image compression in "An Efficient Traffic Image Compression Network".
![eticn](assets/eticn_model.png)
>  The architecture of eticn model.

## Installation
Install CompressAI and the packages required for development.
```bash
conda create -n ETICN python=3.10
conda activate ETICN
pip install pybind11
pip install compressai
git clone https://gitee.com/Hz092811/cgvq-vae-compress.git
cd eticn
pip install -e .
pip install -r requirement.txt
```

## Usage
### Dataset
TODO

### Trainning
An examplary training script with a rate-distortion loss is provided in train.py.
You can adjust the model parameters in cof/eticn.yml
```bash
python train.py --model_config_path config/eticn.yml
```

### Evaluation
you can evaluate a trained model on your own dataset, the evaluation script is:
TODO

## Result
### RD curves

![psnr](assets/R_D_PSNR.png)
![mssim](assets/R_D_MSSIM.png)

>  RD curves on [camvid](https://www.kaggle.com/datasets/carlolepelaars/camvid)、[SODA10M](https://soda-2d.github.io/download.html)、[TRANS](TODO)

### Visualization
![visualization01](assets/vis_1.png)
>  Visualization of the reconstructed image of example one.

![visualization02](assets/vis_2.png)
>  Visualization of the reconstructed image of example two.

### Pretrained Models
TODO

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI
 * Swin-Transformer: https://github.com/microsoft/Swin-Transformer
 * STF:https://github.com/Googolxx/STF
 * camvid Images Dataset: https://www.kaggle.com/datasets/carlolepelaars/camvid
 * SODA10M Images Dataset: https://soda-2d.github.io/download.html


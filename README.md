# Welcome to



![deepflash2](https://raw.githubusercontent.com/matjesg/deepflash2/master/nbs/media/logo/deepflash2_logo_medium.png)

Official repository of deepflash2 - a deep-learning pipeline for segmentation of ambiguous microscopic images.

[![PyPI](https://img.shields.io/pypi/v/deepflash2?color=blue&label=pypi%20version)](https://pypi.org/project/deepflash2/#description) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/deepflash2)](https://pypistats.org/packages/deepflash2)
[![Conda (channel only)](https://img.shields.io/conda/vn/matjesg/deepflash2?color=seagreen&label=conda%20version)](https://anaconda.org/matjesg/deepflash2)
[![GitHub stars](https://img.shields.io/github/stars/matjesg/deepflash2?style=social)](https://github.com/matjesg/deepflash2/)
[![GitHub forks](https://img.shields.io/github/forks/matjesg/deepflash2?style=social)](https://github.com/matjesg/deepflash2/)
***

__The best of two worlds:__
Combining state-of-the-art deep learning with a barrier free environment for life science researchers. 
> Read the [paper](https://arxiv.org/abs/2111.06693), watch the [tutorials](https://matjesg.github.io/deepflash2/tutorial.html), or read the [docs](https://matjesg.github.io/deepflash2/).    
- **No coding skills required** (graphical user interface)
- **Ground truth estimation** from the annotations of multiple experts for model training and validation
- **Quality assurance and out-of-distribution detection** for reliable prediction on new data 
- **Best-in-class performance** for semantic and instance segmentation

<img src="https://github.com/matjesg/deepflash2/blob/master/nbs/media/sample_images.png?raw=true" width="800px" style="max-width: 800pxpx">


<img style="float: left;padding: 0px 10px 0px 0px;" src="https://www.kaggle.com/static/images/medals/competitions/goldl@1x.png">

**Kaggle Gold Medal and Innovation Price Winner:** The *deepflash2* Python API built the foundation for winning the [Innovation Award](https://hubmapconsortium.github.io/ccf/pages/kaggle.html) a Kaggle Gold Medal in the [HuBMAP - Hacking the Kidney](https://www.kaggle.com/c/hubmap-kidney-segmentation) challenge. 
Have a look at our [solution](https://www.kaggle.com/matjes/hubmap-deepflash2-judge-price)

## Quick Start and Demo
> Get started in less than a minute. Watch the <a href="https://matjesg.github.io/deepflash2/tutorial.html" target="_blank">tutorials</a> for help.
#### Demo on Hugging Face Spaces

Go to the [demo space](https://huggingface.co/spaces/matjesg/deepflash2) -- inference only (no training possible).

#### Demo usage with Google Colab

For a quick start, run *deepflash2* in Google Colaboratory (Google account required).

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/deepflash2_GUI.ipynb) 
<video src="https://user-images.githubusercontent.com/13711052/139751414-acf737db-2d8a-4203-8a34-7a38e5326b5e.mov" controls width="100%"></video>

The GUI provides a build-in use for our [sample data](https://github.com/matjesg/deepflash2/releases/tag/sample_data).

1. Starting the GUI (in <a href="https://colab.research.google.com/github/matjesg/deepflash2/blob/master/deepflash2_GUI.ipynb" target="_blank">Colab</a> or follow the installation instructions below)
2. Select the task (GT Estimation, Training, or Prediction) 
3. Click the `Load Sample Data` button in the sidebar and continue to the next sidebar section.

For futher instructions watch the [tutorials](https://matjesg.github.io/deepflash2/tutorial.html).

We provide an overview of the tasks below:

|  | Ground Truth (GT) Estimation | Training | Prediction |
|---|---|---|---|
| Main Task | STAPLE or Majority Voting | Ensemble training  and validation | Semantic and instance segmentation |
| Sample Data | 5 masks from 5 experts each | 5 image/mask pairs | 5 images and 2 trained models |
| Expected Output | 5 GT Segmentation Masks | 5 models | 5 predicted segmentation masks  (semantic and instance) and uncertainty maps|
| Estimated Time | ~ 1 min | ~ 150 min | ~ 4 min |

Times are estimated for Google Colab (with free NVIDIA Tesla K80 GPU).

## Paper and Experiments 

We provide a complete guide to reproduce our experiments using the *deepflash2 Python API* [here](https://github.com/matjesg/deepflash2/tree/master/paper). The data is currently available on [Google Drive](https://drive.google.com/drive/folders/1r9AqP9qW9JThbMIvT0jhoA5mPxWEeIjs?usp=sharing).

The preprint of our paper is available on [arXiv](https://arxiv.org/abs/2111.06693). Please cite

```
@misc{griebel2021deepflash2,
    title={Deep-learning in the bioimaging wild: Handling ambiguous data with deepflash2}, 
    author={Matthias Griebel and Dennis Segebarth and Nikolai Stein and Nina Schukraft and Philip Tovote and Robert Blum and Christoph M. Flath},
    year={2021},
    eprint={2111.06693},
    archivePrefix={arXiv}
}
```



## System requirements
> Works in the browser or on your local pc/server

*deepflash2* is designed to run on Windows, Linux, or Mac (x86-64) if [pytorch](https://pytorch.org/get-started/locally/) is installable.
We generally recommend using Google Colab as it only requires a Google Account and a device with a web browser. 
To run *deepflash2* locally, we recommend using a system with a GPU (e.g., 2 CPUs, 8 GB RAM, NVIDIA GPU with 8GB VRAM or better).

*deepflash2* requires Python>3.6 and the software dependencies are defined in the [settings.ini](https://github.com/matjesg/deepflash2/blob/master/settings.ini) file. Additionally, the ground truth estimation functionalities are based on simpleITK>=2.0 and the instance segmentation capabilities are complemented using cellpose v0.6.6.dev13+g316927e.

*deepflash2* is tested on Google Colab (Ubuntu 18.04.5 LTS) and locally (Ubuntu 20.04 LTS, Windows 10, MacOS 12.0.1).

## Installation Guide
> Typical install time is about 1-5 minutes, depending on your internet connection

The GUI of *deepflash2* runs as a web application inside a Jupyter Notebook, the de-facto standard of computational notebooks in the scientific community. The GUI is built on top of the *deepflash2* Python API, which can be used independently (read the [docs](https://matjesg.github.io/deepflash2/)).
### Google Colab

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/deepflash2_GUI.ipynb) 

Open <a href="https://colab.research.google.com/github/matjesg/deepflash2/blob/master/deepflash2_GUI.ipynb" target="_blank">Colab</a> and excute the `Set up environment` cell or follow the `pip` instructions. Colab provides free access to graphics processing units (GPUs) for fast model training and prediction (Google account required).


### Other systems

We recommend installation into a clean Python 3.7, 3.8, or 3.9 environment (e.g., using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).

#### [mamba](https://github.com/mamba-org/mamba)/[conda](https://docs.conda.io/en/latest/)

Installation with mamba (installaton [instructions](https://github.com/mamba-org/mamba)) allows a fast and realiable installation process (you can replace `mamba` with `conda` and add the `--update-all` flag to do the installation with conda).

```bash
mamba install -c fastchan -c conda-forge -c matjesg deepflash2 
```

#### [pip](https://pip.pypa.io/en/stable/)

If you want to use your GPU and install with pip, we recommend installing PyTorch first by following the [installation instructions](https://pytorch.org/get-started/locally/).

```bash
pip install -U deepflash2
```

#### Using the GUI

If you want to use the GUI, make sure to download the GUI notebook, e.g., using `curl` 

```bash
curl -o deepflash2_GUI.ipynb https://raw.githubusercontent.com/matjesg/deepflash2/master/deepflash2_GUI.ipynb
```

and start a Jupyter server.

```bash
jupyter notebook
```

Then, open `deepflash2_GUI.ipynb` within Notebook environment.

### Docker

Docker images for __deepflash2__ are built on top of [the latest pytorch image](https://hub.docker.com/r/pytorch/pytorch/). 

- CPU only
> `docker run -p 8888:8888 matjes/deepflash2 ./run_jupyter.sh`
- For training, we recommend to run docker with GPU support (You need to install [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) to enable gpu compatibility with these containers.)
> `docker run --gpus all --shm-size=256m -p 8888:8888 matjes/deepflash2 ./run_jupyter.sh`

All docker containers are configured to start a jupyter server. To add data, we recomment using [bind mounts](https://docs.docker.com/storage/bind-mounts/) with `/workspace` as target. To start the GUI, open `deepflash2_GUI.ipynb` within Notebook environment.

For more information on how to run docker see [docker orientation and setup](https://docs.docker.com/get-started/).

## Creating segmentation masks with Fiji/ImageJ

If you don't have labelled training data available, you can use this [instruction manual](https://github.com/matjesg/DeepFLaSH/raw/master/ImageJ/create_maps_howto.pdf) for creating segmentation maps.
The ImagJ-Macro is available [here](https://raw.githubusercontent.com/matjesg/DeepFLaSH/master/ImageJ/Macro_create_maps.ijm).

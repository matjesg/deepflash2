# Welcome to 



![deepflash2](https://raw.githubusercontent.com/matjesg/deepflash2/master/nbs/media/logo/logo_deepflash2_transp-02.png)

Official repository of deepflash2 - a deep learning pipeline for segmentation of fluorescent labels in microscopy images.

![CI](https://github.com/matjesg/deepflash2/workflows/CI/badge.svg) 
[![PyPI](https://img.shields.io/pypi/v/deepflash2?color=blue&label=pypi%20version)](https://pypi.org/project/deepflash2/#description) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/deepflash2)](https://pypistats.org/packages/deepflash2)
[![Conda (channel only)](https://img.shields.io/conda/vn/matjesg/deepflash2?color=seagreen&label=conda%20version)](https://anaconda.org/matjesg/deepflash2) 
[![Build fastai images](https://github.com/matjesg/deepflash2/workflows/Build%20deepflash2%20images/badge.svg)](https://github.com/matjesg/deepflash2)
[![GitHub stars](https://img.shields.io/github/stars/matjesg/deepflash2?style=social)](https://github.com/matjesg/deepflash2/)
[![GitHub forks](https://img.shields.io/github/forks/matjesg/deepflash2?style=social)](https://github.com/matjesg/deepflash2/)
***

## Quick Start in 30 seconds

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/nbs/deepflash2.ipynb)

![deepflash2 training getting started](https://raw.githubusercontent.com/matjesg/deepflash2/master/nbs/media/screen_captures/GUI_Train_start.gif)

Examplary training workflow.

## Why using deepflash2?

__The best of two worlds:__
Combining state of the art deep learning with a barrier free environment for life science researchers.

- End-to-end process for life science researchers
    - graphical user interface - no coding skills required
    - free usage on _Google Colab_ at no costs
    - easy deployment on own hardware
- Rigorously evaluated deep learning models
    - Model Library
    - easy integration new (*pytorch*) models
- Best practices model training
    - leveraging the _fastai_ library
    - mixed precision training
    - learning rate finder and fit one cycle policy 
    - advanced augementation 
- Reliable prediction on new data
    - leveraging Bayesian Uncertainties

## Workflow
## Using Google Colab:

Note: If not already done, we recommend you to read the “Before you get started guide” first. 

For using deepflash2 in Google Colab follow this link: (https://colab.research.google.com/github/matjesg/deepflash2/blob/master/nbs/deepflash2.ipynb)
	
---

### 1.0 Setup Environment: 

At first you must allow google colab to run the notebook from GitHub by accepting the prompt.

---

### 2.0 Connect your google drive:

It is recommended to connect your runtime to google drive.
You can allow it with ‘y’ for yes or deny it with ‘n’ for no.

Go to the Url that is presented to you in the “Set up environment cell”.
There you can choose your google drive account to connect to google colab.

After you have successfully connected the accounts, google will present you a one time authorization code. Copy this code and enter it in the according field in the “Set up environment cell” and continue by pressing the enter key.

Note: The authentication key will only work with this runtime. When you close google colab you have to request a new code as described.

---

### 3.0 Start deepflash2 UI:

Note: You have to create a Folder Structure as described here, before starting the program.

When you run this cell, the UI opens.
At first click “Select Project Folder”. 
This unfolds your google drive main directory.
There you can browse through the folders and select the correct folders.
After that hit Save. Now the selected folder is connected to deepFlash2.
The select folder tab will change to the name of the selected folder.
If you want to change the folder, click this button again.

---

### 4.0 GT Estimation

### 4.1. Expert Annotations:

You can estimate the ground truth by selecting images that are segmented beforehand by different experts. We recommend at least 12 different images from 3 different experts. After you have selected the desired data, you can press Load Data. 
Also you have the possibility to use sample data by clicking on “Load Sample Data”. There are 5 times 5 pictures available, coming from different experts.

### 4.2. Ground Truth Estimation:

When you have uploaded the images you can start the ground truth estimation by selecting one of the presented algorithms.
At the time of writing you can select between STAPLE and Majority Voting.
We recommend the STAPLE Algorithm when:
- if the experience of experts that have annotated the images vary or is unknown
- if you need more precise results when compared with the Majority Voting Algorithm

We recommend the Majority Voting Algorithm when:
- Use this algorithm if you can be sure that the expert annotations have no repeated errors.

When the estimation is finished you can download the ground truth images for further use in training.

---

### 5.0 Training the model

Note: For comprehensive results you require a ground truth image that is according to the content found in the training images you want to use to train the neural network. E.G. if you want to train a neural network to find specific objects in the images, these objects should also be found in the expert images used for the ground truth estimation.

In this step you will create a model that you can use to automatically annotate new images.

### 5.1 Data

First, you have to provide training images. These should be unsegmented and contain the objects you want the neural network to find.
Second, you have to offer segmentation masks you have to create beforehand.
Third, you have to select a number of classes. We recommend X.
Fourth, you can provide instance labels. This step is optional.

### 5.2 Ensemble Training

Note: We recommend that you reserve 70% of your images data for model training.

You can use the Ensemble training to optimize the results.
First choose a Number of models within an ensemble; If you're experimenting with parameters, try only one model first; Depending on the data, ensembles should at least comprise 3-5 models
How many times a single model is trained on a mini-batch of the training data. 

Train all models (ensemble) or (re-)train specific model.

### 5.3 Validation

Note: We recommend that you reserve 30% of your image data for validation.

Here you can validate the performance of the model you have trained before with unsegmented images.

---


### 6.0 Prediction

In this section you can use a model to segment the images and evaluate the results.

### 6.1 Data and Ensemble

First you have to upload the images you want the model to work with and predict results.
Second you have to select the model you want to apply.

### 6.2 Prediction and Quality Control

Here you can run the prediction and download the results.
You can enable test-time augmentation for prediction (more reliable and accurate, but slow).

---

If errors occur at any point, refer to the “common problems” section for help. 



## Workflow

tbd

## Installing

You can use **deepflash2** by using [Google Colab](https://research.google.com/colaboratory/). You can run every page of the [documentation](https://matjesg.github.io/deepflash2/) as an interactive notebook - click "Open in Colab" at the top of any page to open it.
 - Be sure to change the Colab runtime to "GPU" to have it run fast!
 - Use Firefox or Google Chrome if you want to upload your images.

You can install **deepflash2**  on your own machines with conda (highly recommended):

```bash
conda install -c fastai -c pytorch -c matjesg deepflash2 
```
To install with pip, use

```bash
pip install deepflash2
```
If you install with pip, you should install PyTorch first by following the PyTorch [installation instructions](https://pytorch.org/get-started/locally/).

```python
tbd
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-57f17817c86e> in <module>
    ----> 1 tbd
    

    NameError: name 'tbd' is not defined


## Using Docker

Docker images for __deepflash2__ are built on top of [the latest pytorch image](https://hub.docker.com/r/pytorch/pytorch/) and [fastai](https://github.com/fastai/docker-containers) images. **You must install [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) to enable gpu compatibility with these containers.**

- CPU only
> `docker run -p 8888:8888 matjesg/deepflash`
- With GPU support ([Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) must be installed.)
has an editable install of fastai and fastcore.
> `docker run --gpus all -p 8888:8888 matjesg/deepflash`
All docker containers are configured to start a jupyter server. **deepflash2** notebooks are available in the `deepflash2_notebooks` folder.

For more information on how to run docker see [docker orientation and setup](https://docs.docker.com/get-started/) and [fastai docker](https://github.com/fastai/docker-containers).

## Model Library

We provide a model library with pretrained model weights. Visit our [model library documentation](https://matjesg.github.io/deepflash2/model_library.html) for information on the datasets of the pretrained models.

## Creating segmentation masks with Fiji/ImageJ

If you don't have labelled training data available, you can use this [instruction manual](https://github.com/matjesg/DeepFLaSH/raw/master/ImageJ/create_maps_howto.pdf) for creating segmentation maps.
The ImagJ-Macro is available [here](https://raw.githubusercontent.com/matjesg/DeepFLaSH/master/ImageJ/Macro_create_maps.ijm).

## Acronym

A Deep-learning pipeline for Fluorescent Label Segmentation that learns from Human experts

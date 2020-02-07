# DeepFLaSH2
> Official repository of DeepFLasH - a deep learning pipeline for segmentation of fluorescent labels in microscopy images.


This file will become your README and also the index of your documentation.

## Install

`pip install deepflash2`

## How to use

```python
learn = Unet_Learner(train_generator, valid_generator)
```

```python
learn.lr_find()
learn.plot_loss()
```

```python
learn.fit_one_cycle(100, validation_freq=5, max_lr=5e-4)
```

## Model Library

This list contains download links to the weights of the selected models as well as an example of their corresponding training images and masks.

You can select and apply these models within our Jupyter Notebook.

## Acronym

A Deep-learning pipeline for Fluorescent Label Segmentation that learns from Human experts

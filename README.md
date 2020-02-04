# DeepFLaSH2
> Official repository of DeepFLasH - a deep learning pipeline for segmentation of fluorescent labels in microscopy images..


This file will become your README and also the index of your documentation.

## Install

`pip install deepflash2`

## How to use

Fill me in please! Don't forget code examples:

```python
unet = Unet2D()
```

```python
losses = {'conv_u0d-score':weighted_softmax_cross_entropy, 
          'softmax':zero_loss}
metrics = {'softmax': [tf.keras.metrics.Recall(class_id=1), 
                       tf.keras.metrics.Precision(class_id=1),
                       tf.keras.metrics.BinaryAccuracy(),
                       IoU(num_classes=2, class_id=1, name='IoU')
                      ]}

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

unet.model.compile(optimizer=opt, loss=losses, metrics=metrics)
```

## Model Library

This list contains download links to the weights of the selected models as well as an example of their corresponding training images and masks.

You can select and apply these models within our Jupyter Notebook.

## Acronym

A Deep-learning pipeline for Fluorescent Label Segmentation that learns from Human experts

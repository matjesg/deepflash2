# Release notes

<!-- do not remove -->

## 0.1.2

### New Features

- Real-Time loss weight computation and large file support ([#11](https://github.com/matjesg/deepflash2/pull/11)), thanks to [@matjesg](https://github.com/matjesg)
  - - Real-Time loss weight computation via fast convolutional distance transform on GPU
- temp storage using zarr instead of RAM
- zarr dependency

### Bugs Squashed

- Prediction not possible on Windows machines due to cuda error ([#10](https://github.com/matjesg/deepflash2/issues/10))
  - During prediction on a windows machine a a OS Error occurs due to: 

"RuntimeError: cuda runtime error (801) : operation not supported at ..\torch/csrc/generic/StorageSharing.cpp:247"

Problem: Storage sharing currently not supported on windows.

Proposed solution: Ensemble learner takes "num_workers" argument and passes it to subsequent functions. If num_workers == 0, prediction works for me.


## 0.1.1

- Bud fixes and minor improvements.

## 0.1.0

### New Features

- Adding GUI and new project structure ([#8](https://github.com/matjesg/deepflash2/pull/8)), thanks to [@matjesg](https://github.com/matjesg)

## 0.0.14

### New Features

- Adding test time augmentation ([#7](https://github.com/matjesg/deepflash2/pull/7)), thanks to [@matjesg](https://github.com/matjesg)

### Bugs Squashed

- Type Error when starting training ([#5](https://github.com/matjesg/deepflash2/issues/5))
  - When I try to start the training process in google colab, this error occurs:

**TypeError: no implementation found for 'torch.nn.functional.cross_entropy' on types that implement __torch_function__: [<class 'fastai.torch_core.TensorImage'>, <class 'fastai.torch_core.TensorMask'>]**

as well as 

**FileNotFoundError: [Errno 2] No such file or directory: 'models/model.pth'**

in the end.


Hope you know whats the problem.


## 0.0.13


### Bugs Squashed

- Type Error when starting training ([#5](https://github.com/matjesg/deepflash2/issues/5))
  - When I try to start the training process in google colab, this error occurs:

**TypeError: no implementation found for 'torch.nn.functional.cross_entropy' on types that implement __torch_function__: [<class 'fastai.torch_core.TensorImage'>, <class 'fastai.torch_core.TensorMask'>]**

as well as 

**FileNotFoundError: [Errno 2] No such file or directory: 'models/model.pth'**

in the end.


Hope you know whats the problem.


## 0.0.12


### Bugs Squashed

- Checking for SimpleITK 1.2.4 and install if not available ([#4](https://github.com/matjesg/deepflash2/pull/4)), thanks to [@matjesg](https://github.com/matjesg)
  - Closes #3


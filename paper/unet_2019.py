"""
unet_2019.py

Code slightly adapted from:

Falk, Thorsten, et al.
"U-Net: deep learning for cell counting, detection, and morphometry." 
Nature methods 16.1 (2019): 67-70.
"""



import keras
import keras.layers
import keras.optimizers
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np
import math, random
from time import time
from scipy import ndimage
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d

'''
Deformation field class for data augmentation
'''
class DeformationField:

  def __init__(self, shape=(540,540)):
    self.shape = shape
    self.deformationField = np.meshgrid(
      *[np.arange(d) - d / 2 for d in shape])[::-1]

  def rotate(self, theta=0, phi=0, psi=0):
    if len(self.shape) == 2:
      self.deformationField = [
        self.deformationField[0] * math.cos(theta) +
        self.deformationField[1] * math.sin(theta),
        -self.deformationField[0] * math.sin(theta) +
        self.deformationField[1] * math.cos(theta)]
    else:
      self.deformationField = [
        self.deformationField[0],
        self.deformationField[1] * math.cos(theta) +
        self.deformationField[2] * math.sin(theta),
        -self.deformationField[1] * math.sin(theta) +
        self.deformationField[2] * math.cos(theta)]
      self.deformationField = [
        self.deformationField[0] * math.cos(phi) +
        self.deformationField[2] * math.sin(phi),
        self.deformationField[1]
        -self.deformationField[0] * math.sin(phi) +
        self.deformationField[2] * math.cos(phi)]
      self.deformationField = [
        self.deformationField[0],
        self.deformationField[1] * math.cos(psi) +
        self.deformationField[2] * math.sin(psi),
        -self.deformationField[1] * math.sin(psi) +
        self.deformationField[2] * math.cos(psi)]

  def mirror(self, dims):
    for d in range(len(self.shape)):
      if dims[d]:
        self.deformationField[d] = -self.deformationField[d]

  def addRandomDeformation(self, grid=(150, 150), sigma=(10, 10)):
    seedGrid = np.meshgrid(
      *[np.arange(-g / 2, s + g / 2, g) for (g, s) in zip(grid, self.shape)])
    seed = [np.random.normal(0, s, g.shape) for (g, s) in zip(seedGrid, sigma)]
    defFcn = [Rbf(*seedGrid, s, function="cubic") for s in seed]
    targetGrid = np.meshgrid(*map(np.arange, self.shape))
    deformation = [f(*targetGrid) for f in defFcn]
    self.deformationField = [
      f + df for (f, df) in zip(self.deformationField, deformation)]

  def get(self, offset=(0, 0), pad=(0, 0)):
    sliceDef = tuple(slice(int(p/2),int(-p/2)) if p > 0 else None for p in pad)
    deform = [d[sliceDef] for d in self.deformationField]
    return [d + offs for (d, offs) in zip(deform, offset)]

  def apply(self, data, offset=(0, 0), pad=(0, 0), order=1):
    coords = [d.flatten() for d in self.get(offset, pad)]
    outshape = tuple(int(s - p) for (s, p) in zip(self.shape, pad))
    if len(data.shape) == len(self.shape) + 1:
      tile = np.empty((*outshape, data.shape[-1]))
      for c in range(data.shape[-1]):
        tile[...,c] = ndimage.interpolation.map_coordinates(
          data[...,c], coords, order=order, mode="reflect").reshape(
            outshape)
      return tile.astype(data.dtype)
    else:
      return ndimage.interpolation.map_coordinates(
        data, coords, order=order, mode="reflect").reshape(outshape).astype(
          data.dtype)

class DataPreProcessor:

  def __init__(self, element_size_um=None,
               border_weight_sigma_px=6, border_weight_factor=50,
               foreground_background_ratio=1.0):
    self.element_size_um = element_size_um
    self.border_weight_sigma_px = border_weight_sigma_px
    self.border_weight_factor = border_weight_factor
    self.foreground_background_ratio = foreground_background_ratio

  def generateSample(
      self, data, instancelabels=None, classlabels=None, ignore=None,
      weights=None):

    dataScaled = data["rawdata"]
    elSize = data["element_size_um"]
    nDims = len(dataScaled.shape) - 1
    instlabels = instancelabels
    clabels = classlabels
    ign = ignore
    wghts = weights

    # If weights need to be computed, and no instance labels are given,
    # generate them now
    if wghts is None and clabels is not None and instlabels is None:
      instlabels = np.zeros_like(clabels)
      classes = np.unique(clabels)[1:]
      nextInstance = 1
      for c in classes:
        comps, nInstances = ndimage.measurements.label(clabels == c)
        instlabels[comps > 0] = comps[comps > 0] + nextInstance
        nextInstance += nInstances

    # Rescale blobs to processing element size
    if self.element_size_um is not None and np.any(
        np.asarray(elSize) != np.asarray(self.element_size_um)):
      print("  Rescaling...")
      scales = tuple(s / t for (s, t) in zip(elSize, self.element_size_um))

      dataScaled = ndimage.zoom(
        dataScaled, (*scales, 1), order=1, mode="reflect")

      if instlabels is not None:
        instlabels = ndimage.zoom(
          instlabels, scales, order=0, mode="reflect")
      if clabels is not None:
        clabels = ndimage.zoom(
          clabels, scales, order=0, mode="reflect")
      if ign is not None:
        ign = ndimage.zoom(
          ign, scales, order=0, mode="reflect")
      if wghts is not None:
        wghts = ndimage.zoom(
          wghts, scales, order=1, mode="reflect")

    # Normalize values to [0,1] range
    print("  Normalizing intensity range...")
    for c in range(dataScaled.shape[-1]):
      minValue = np.min(dataScaled[...,c])
      maxValue = np.max(dataScaled[...,c])
      dataScaled[...,c] = (dataScaled[...,c] - minValue) / (maxValue - minValue)

    # If no labels are given we are done and simply return the data array
    if instlabels is None and clabels is None:
      return dataScaled.astype(np.float32), None, None, None

    # If no classlabels are given treat the problem as binary segmentation
    # ==> Create a new array assigning class 1 (foreground) to each instance
    if clabels is None:
      clabels = instlabels > 0

    # If weights are given we only need to compute the sample pdf and we're
    # done
    if wghts is not None:
      pdf = (clabels > 0) + self.foreground_background_ratio * (
        clabels == 0)
      if ign is not None:
        pdf *= 1 - ign
      return (dataScaled.astype(np.float32), clabels.astype(np.int32),
              wghts.astype(np.float32), pdf.astype(np.float32))

    # No weights given, so we need to compute them

    # Initialize label and weights arrays with background
    labels = np.zeros_like(clabels)
    wghts = self.foreground_background_ratio * np.ones_like(clabels)

    # Get all foreground class labels
    classes = np.unique(clabels)[1:]

    for c in classes:

      # Extract all instance labels of class c
      instances = np.unique(instlabels * (clabels == c))[1:]

      # Generate background ridges between touching instances
      # of that class
      print("  Generating ridges...")
      for instance in instances:
        objectMaskDil = ndimage.morphology.binary_dilation(
          labels == c, structure=np.ones((3,) * nDims))
        labels[(instlabels == instance) & (objectMaskDil == 0)] = c

      # Generate weights
      print("   Generating weights...")
      min1dist = 1e10 * np.ones(labels.shape)
      min2dist = 1e10 * np.ones(labels.shape)
      for instance in instances:
        dt = ndimage.morphology.distance_transform_edt(
          instlabels != instance)
        min2dist = np.minimum(min2dist, dt)
        newMin1 = np.minimum(min1dist, min2dist)
        newMin2 = np.maximum(min1dist, min2dist)
        min1dist = newMin1
        min2dist = newMin2
      wghts += self.border_weight_factor * np.exp(
        -(min1dist + min2dist) * (min1dist + min2dist) /
        (self.border_weight_sigma_px * self.border_weight_sigma_px));

    # Set foreground weights to 1
    wghts[labels > 0] = 1
    pdf = (labels > 0) + (labels == 0) * self.foreground_background_ratio

    # Set weight and sampling probability for ignored regions to 0
    if ign is not None:
      wghts[ign] = 0
      pdf[ign] = 0

    return (dataScaled.astype(np.float32), labels.astype(np.int32),
            wghts.astype(np.float32), pdf.astype(np.float32))

'''
Data Augmentation Generator for U-Net input
'''
class DataAugmentationGenerator(keras.utils.Sequence):

  '''
    data - A list of tuples of the form
           [{ rawdata: numpy.ndarray (HxWxC),
              element_size_um: [e_y, e_x] }, ...]
           containing the raw data ([0-1] normalized) and corresponding
           element sizes in micrometers
    instancelabels - A list containing the corresponding instance labels.
                     0 = background, 1-m instance labels
    tile_shape - The tile shape the network expects as input
    padding - The padding (input shape - output shape)
    classlabels - A list containing the corresponding class labels.
                  0 = ignore, 1 = background, 2-n foreground classes
                  If None, the problem will be treated as binary segmentation
    n_classes - The number of classes including background
    ignore - A list containing the corresponding ignore regions.
    weights - A list containing the corresponding weights.
    element_size_um - The target pixel size in micrometers
    batch_size - The number of tiles to generate per batch
    rotation_range_deg - (alpha_min, alpha_max): The range of rotation angles.
                         A random rotation is drawn from a uniform distribution
                         in the given range
    flip - If true, a coin flip decides whether a mirrored tile will be
           generated
    deformation_grid - (dx, dy): The distance of neighboring grid points in
                       pixels for which random deformation vectors are drawn
    deformation_magnitude - (sx, sy): The standard deviations of the
                            Gaussians, the components of the deformation
                            vector are drawn from
    value_minimum_range - (v_min, v_max): Input intensity zero will be mapped
                          to a random value in the given range
    value_maximum_range - (v_min, v_max): Input intensity one will be mapped
                          to a random value within the given range
    value_slope_range - (s_min, s_max): The slope at control points is drawn
                        from a uniform distribution in the given range
    border_weight_sigma_px - The border weight standard deviation in pixels
    border_weight_factor - The border weight factor to enforce instance
                           separation
    foreground_background_ratio - The ratio between foreground and background
                                  pixels
  '''
  def __init__(self, data, tile_shape, padding, instancelabels=None,
               classlabels=None, n_classes=2, ignore=None, weights=None,
               batch_size = 1, element_size_um=None, rotation_range_deg=(0, 0),
               flip=None, deformation_grid=None, deformation_magnitude=(0, 0),
               value_minimum_range=(0, 0), value_maximum_range=(1, 1),
               value_slope_range=(1, 1), shuffle=True,
               border_weight_sigma_px=6, border_weight_factor=50,
               foreground_background_ratio=0.1):

    assert(instancelabels is not None or classlabels is not None)

    self.tile_shape = tile_shape
    self.padding = padding
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.rotation_range_deg = rotation_range_deg
    self.flip = flip
    self.deformation_grid = deformation_grid
    self.deformation_magnitude = deformation_magnitude
    self.value_minimum_range = value_minimum_range
    self.value_maximum_range = value_maximum_range
    self.value_slope_range = value_slope_range
    self.shuffle = shuffle

    self.n_channels = data[0]["rawdata"].shape[2]
    self.output_shape = tuple(int(t - p) for (t, p) in zip(tile_shape, padding))

    pre = DataPreProcessor(
      element_size_um, border_weight_sigma_px, border_weight_factor,
      foreground_background_ratio)

    self.data = []
    self.labels = []
    self.weights = []
    self.pdf = []
    for i in range(len(data)):

      print("Processing training sample {} / {}".format(i+1, len(data)))

      (sampleData, sampleLabels, sampleWeights, samplePdf) = pre.generateSample(
        data[i], instancelabels[i] if instancelabels is not None else None,
        classlabels=classlabels[i] if classlabels is not None else None,
        ignore=ignore[i] if ignore is not None else None,
        weights=weights[i] if weights is not None else None)
      self.data.append(sampleData)
      self.labels.append(sampleLabels)
      self.weights.append(sampleWeights)
      self.pdf.append(samplePdf)

    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.data) / self.batch_size))

  def __getitem__(self, index):
    return self.__data_generation(
      self.indexes[index*self.batch_size:(index+1)*self.batch_size])

  def on_epoch_end(self):
    print()
    self.indexes = np.arange(len(self.data))

    if self.shuffle:
      #print("Shuffling training samples")
      np.random.shuffle(self.indexes)

    #print("Generating deformation field")
    self.deformationField = DeformationField(self.tile_shape)

    if self.rotation_range_deg[1] > self.rotation_range_deg[0]:
      self.deformationField.rotate(
        theta=math.pi *
        (np.random.random() *
         (self.rotation_range_deg[1] - self.rotation_range_deg[0]) +
         self.rotation_range_deg[0]) / 180.0)

    if self.flip:
      self.deformationField.mirror(
        (random.randint(0, 1) == 1, random.randint(0, 1) == 1))

    if self.deformation_grid is not None:
      self.deformationField.addRandomDeformation(
        self.deformation_grid, self.deformation_magnitude)

    #print("Generating value augmentation function")
    minValue = self.value_minimum_range[0] + (
      self.value_minimum_range[1] -
      self.value_minimum_range[0]) * np.random.random()
    maxValue = self.value_maximum_range[0] + (
      self.value_maximum_range[1] -
      self.value_maximum_range[0]) * np.random.random()
    intermediateValue = 0.5 * (
      self.value_slope_range[0] + (
        self.value_slope_range[1] - self.value_slope_range[0]) *
      np.random.random())
    self.gammaFcn = interp1d(
      [0, 0.5, 1.0], [minValue, intermediateValue, maxValue], kind='quadratic')

  def __data_generation(self, indexes):
    X = np.empty((self.batch_size, *self.tile_shape, self.n_channels),
                 dtype=np.float32)
    Y = np.zeros((self.batch_size, *self.output_shape), dtype=np.int)
    W = np.empty((self.batch_size, *self.output_shape), dtype=np.float)
    for i, idx in enumerate(indexes):
      cumulatedPdf = np.cumsum(self.pdf[idx] / np.sum(self.pdf[idx]))
      center = np.unravel_index(
        np.argmax(cumulatedPdf > np.random.random()), self.pdf[idx].shape)
      X[i,...] = self.gammaFcn(self.deformationField.apply(
        self.data[idx], center).flatten()).reshape(
          (*self.tile_shape, self.n_channels))
      Y[i,...] = self.deformationField.apply(
        self.labels[idx], center, self.padding, 0)
      W[i,...] = self.deformationField.apply(
        self.weights[idx], center, self.padding, 1)

    return {
      "data": X,
      "labels": keras.utils.to_categorical(Y, num_classes=self.n_classes),
      "weights": W }, None

'''
Tile Generator for U-Net input
'''
class TileGenerator(keras.utils.Sequence):

  '''
    data - A list of tuples of the form
           [{ rawdata: numpy.ndarray (HxWxC),
              element_size_um: [e_y, e_x] }, ...]
           containing the raw data ([0-1] normalized) and corresponding
           element sizes in micrometers
    instancelabels - A list containing the corresponding instance labels.
                     0 = background, 1-m instance labels
    tile_shape - The tile shape the network expects as input
    padding - The padding (input shape - output shape)
    classlabels - A list containing the corresponding class labels.
                   0 = ignore, 1 = background, 2-n foreground classes
                   If None, the problem will be treated as binary segmentation
    n_classes - The number of classes including background
    ignore - A list containing the corresponding ignore regions.
    weights - A list containing the corresponding weights.
    element_size_um - The target pixel size in micrometers
    border_weight_sigma_px - The border weight standard deviation in pixels
    border_weight_factor - The border weight factor to enforce instance
                           separation
    foreground_background_ratio - The ratio between foreground and background
                                  pixels
  '''
  def __init__(self, data, tile_shape, padding, instancelabels=None,
               classlabels=None, n_classes=2, ignore=None, weights=None,
               element_size_um=None, border_weight_sigma_px=6,
               border_weight_factor=50, foreground_background_ratio=0.1):
    self.tile_shape = tile_shape
    self.padding = padding
    self.n_classes = n_classes

    self.n_channels = data[0]["rawdata"].shape[-1]
    self.output_shape = tuple(int(t - p) for (t, p) in zip(tile_shape, padding))

    pre = DataPreProcessor(
      element_size_um, border_weight_sigma_px, border_weight_factor,
      foreground_background_ratio)
    tiler = DeformationField(tile_shape)

    self.hasLabels = instancelabels is not None or classlabels is not None
    self.data = []
    self.labels = [] if self.hasLabels else None
    self.weights = [] if self.hasLabels else None
    self.image_indices = []
    self.image_shapes = []
    self.in_slices = []
    self.out_slices = []

    for i in range(len(data)):

      print("Processing test sample {} / {}".format(i+1, len(data)))

      (sampleData, sampleLabels, sampleWeights, _) = pre.generateSample(
        data[i], instancelabels[i] if instancelabels is not None else None,
        classlabels=classlabels[i] if classlabels is not None else None,
        ignore=ignore[i] if ignore is not None else None,
        weights=weights[i] if weights is not None else None)

      # Tiling
      print("  Tiling...")
      data_shape = sampleData.shape[:-1]
      for ty in range(int(np.ceil(data_shape[0] / self.output_shape[0]))):
        for tx in range(int(np.ceil(data_shape[1] / self.output_shape[1]))):
          centerPos = (int((ty + 0.5) * self.output_shape[0]),
                       int((tx + 0.5) * self.output_shape[1]))
          self.data.append(tiler.apply(sampleData, centerPos))
          if self.hasLabels:
            self.labels.append(
              tiler.apply(sampleLabels, centerPos, padding, order=0))
            self.weights.append(
              tiler.apply(sampleWeights, centerPos, padding, order=1))
          self.image_indices.append(i)
          self.image_shapes.append(data_shape)
          sliceDef = tuple(
            slice(tIdx * o, min((tIdx + 1) * o, s))
            for (tIdx, o, s) in zip((ty, tx), self.output_shape, data_shape))
          self.out_slices.append(sliceDef)
          sliceDef = tuple(
            slice(0, min((tIdx + 1) * o, s) - tIdx * o)
            for (tIdx, o, s) in zip((ty, tx), self.output_shape, data_shape))
          self.in_slices.append(sliceDef)

    self.on_epoch_end()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.__data_generation(index)

  def on_epoch_end(self):
    pass

  def __data_generation(self, idx):
    X = np.empty((1, *self.tile_shape, self.n_channels), dtype=np.float32)
    Y = np.zeros(
      (1, *self.output_shape), dtype=np.int) if self.hasLabels else None
    W = np.empty(
      (1, *self.output_shape), dtype=np.float) if self.hasLabels else None
    X[0,...] = self.data[idx]
    if self.hasLabels:
      Y[0,...] = self.labels[idx]
      W[0,...] = self.weights[idx]

    return {
      "data": X,
      "labels": keras.utils.to_categorical(
        Y, num_classes=self.n_classes) if Y is not None else None,
      "weights": W,
      "image_index": self.image_indices[idx],
      "image_shape": self.image_shapes[idx],
      "out_slice": self.out_slices[idx],
      "in_slice": self.in_slices[idx] }, None

#
# Finally the U-Net
#

class Unet2D:

  def __init__(self, snapshot=None, n_channels=1, n_classes=2, n_levels=4,
               n_features=64, name="U-Net"):

    self.concat_blobs = []

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.n_levels = n_levels
    self.n_features = n_features
    self.name = name

    self.trainModel, self.padding = self._createModel(True)
    self.testModel, _ = self._createModel(False)

    if snapshot is not None:
      self.trainModel.load_weights(snapshot)
      self.testModel.load_weights(snapshot)

  def _weighted_categorical_crossentropy(self, y_true, y_pred, weights):
    return tf.losses.softmax_cross_entropy(
      y_true, y_pred, weights=weights, reduction=tf.losses.Reduction.MEAN)

  def _createModel(self, training):

    data = keras.layers.Input(shape=(None, None, self.n_channels), name="data")

    concat_blobs = []

    if training:
      labels = keras.layers.Input(
        shape=(None, None, self.n_classes), name="labels")
      weights = keras.layers.Input(shape=(None, None), name="weights")

    # Modules of the analysis path consist of two convolutions and max pooling
    for l in range(self.n_levels):
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 3, padding="valid",
          kernel_initializer="he_normal",
          name="conv_d{}a-b".format(l))(data if l == 0 else t))
      concat_blobs.append(
        keras.layers.LeakyReLU(alpha=0.1)(
          keras.layers.Conv2D(
            2**l * self.n_features, 3, padding="valid",
            kernel_initializer="he_normal", name="conv_d{}b-c".format(l))(t)))
      t = keras.layers.MaxPooling2D(pool_size=(2, 2))(concat_blobs[-1])

    # Deepest layer has two convolutions only
    t = keras.layers.LeakyReLU(alpha=0.1)(
      keras.layers.Conv2D(
        2**self.n_levels * self.n_features, 3, padding="valid",
        kernel_initializer="he_normal",
        name="conv_d{}a-b".format(self.n_levels))(t))
    t = keras.layers.LeakyReLU(alpha=0.1)(
      keras.layers.Conv2D(
        2**self.n_levels * self.n_features, 3, padding="valid",
        kernel_initializer="he_normal",
        name="conv_d{}b-c".format(self.n_levels))(t))
    pad = 8

    # Modules in the synthesis path consist of up-convolution,
    # concatenation and two convolutions
    for l in range(self.n_levels - 1, -1, -1):
      name = "upconv_{}{}{}_u{}a".format(
        *(("d", l+1, "c", l) if l == self.n_levels - 1 else ("u", l+1, "d", l)))
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 2, padding="same",
          kernel_initializer="he_normal", name=name)(
            keras.layers.UpSampling2D(size = (2,2))(t)))
      t = keras.layers.Concatenate()(
        [keras.layers.Cropping2D(cropping=int(pad / 2))(concat_blobs[l]), t])
      pad = 2 * (pad + 8)
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 3, padding="valid",
          kernel_initializer="he_normal", name="conv_u{}b-c".format(l))(t))
      t = keras.layers.LeakyReLU(alpha=0.1)(
        keras.layers.Conv2D(
          2**l * self.n_features, 3, padding="valid",
          kernel_initializer="he_normal", name="conv_u{}c-d".format(l))(t))
    pad /= 2

    score = keras.layers.Conv2D(
      self.n_classes, 1, kernel_initializer = 'he_normal',
      name="conv_u0d-score")(t)
    softmax_score = keras.layers.Softmax()(score)

    if training:
      model = keras.Model(inputs=[data, labels, weights], outputs=softmax_score)
      model.add_loss(
        self._weighted_categorical_crossentropy(labels, score, weights))
      adam = keras.optimizers.Adam(
        lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0,
        amsgrad=False)
      model.compile(optimizer=adam, loss=None)
    else:
      model = keras.Model(inputs=data, outputs=softmax_score)

    return model, int(pad)

  def loadCaffeModelH5(self, path):
    train_layer_dict = dict([(layer.name, layer)
                             for layer in self.trainModel.layers])
    test_layer_dict = dict([(layer.name, layer)
                            for layer in self.testModel.layers])
    pre = h5py.File(path, 'a')
    l = list(pre['data'].keys())
    for i in l:
      kernel = pre['data'][i]['0'][()]
      bias = pre['data'][i]['1'][()]
      train_layer_dict[i].set_weights([kernel,bias])
      test_layer_dict[i].set_weights([kernel,bias])
    pre.close()

  def train(self, sample_generator, validation_generator=None,log_dir='logs',
            validation_freq=50,
            n_epochs=100, snapshot_interval=1, snapshot_prefix=None):

    callbacks = [TensorBoard(log_dir=f"{log_dir}/{self.name}-{time()}")]
    if snapshot_prefix is not None:
      callbacks.append(keras.callbacks.ModelCheckpoint(
        (snapshot_prefix if snapshot_prefix is not None else self.name) +
        ".{epoch:04d}.h5", mode='auto', period=snapshot_interval))
    self.trainModel.fit_generator(
      sample_generator, epochs=n_epochs, validation_data=validation_generator,
      verbose=1, validation_freq=validation_freq, callbacks=callbacks)

  def predict(self, tile_generator):

    smscores = []
    segmentations = []

    for tileIdx in range(tile_generator.__len__()):
      tile = tile_generator.__getitem__(tileIdx)
      outIdx = tile[0]["image_index"]
      outShape = tile[0]["image_shape"]
      outSlice = tile[0]["out_slice"]
      inSlice = tile[0]["in_slice"]
      softmax_score = self.testModel.predict(tile[0]["data"], verbose=1)
      if len(smscores) < outIdx + 1:
        smscores.append(np.empty((*outShape, self.n_classes)))
        segmentations.append(np.empty(outShape))
      smscores[outIdx][outSlice] = softmax_score[0][inSlice]
      segmentations[outIdx][outSlice] = np.argmax(
        softmax_score[0], axis=-1)[inSlice]

    return smscores, segmentations


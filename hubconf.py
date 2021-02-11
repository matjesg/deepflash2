# Optional list of dependencies required by the package
dependencies = ['torch', 'fastai']

from deepflash2.models import unet_deepflash2 as _unet_deepflash2
from deepflash2.models import unext50_deepflash2 as _unext50_deepflash2
from deepflash2.models import unet_falk2019 as _unet_falk2019
from deepflash2.models import unet_ronneberger2015 as _unet_ronneberger2015

def unext50_deepflash2(pretrained=None, **kwargs):
    """
    U-NeXt50 model optimized for deepflash2
    pretrained (str): specifies the dataset for pretrained weights
    """
    model = _unext50_deepflash2(pretrained=pretrained, **kwargs)
    return model

def unet_deepflash2(pretrained=None, **kwargs):
    """
    U-Net model optimized for deepflash2
    pretrained (str): specifies the dataset for pretrained weights
    """
    model = _unet_deepflash2(pretrained=pretrained, **kwargs)
    return model

def unet_falk2019(pretrained=None, **kwargs):
    """
    U-Net model according to Falk, T., D. Mai, R. Bensch, Ö. Çiçek, A. Abdulkadir, Y. Marrakchi, … others. (2019). “U-Net: deep learning for cell counting, detection, and morphometry.” Nature Methods, 16 (1), 67–70.
    pretrained (str): specifies the dataset for pretrained weights
    """
    model = _unet_falk2019(pretrained=pretrained, **kwargs)
    return model

def unet_ronneberger2015(pretrained=None, **kwargs):
    """
    Original U-Net model according to Ronneberger, O., P. Fischer and T. Brox. (2015). “U-net: Convolutional networks for biomedical image segmentation.” In: International Conference on Medical image computing and computer-assisted intervention (pp. 234–241).
    pretrained (str): specifies the dataset for pretrained weights
    """
    model = _unet_ronneberger2015(pretrained=pretrained, **kwargs)
    return model
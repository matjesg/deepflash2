# Optional list of dependencies required by the package
dependencies = ['torch']

from deepflash2.models import unet_deepflash2 as _unet_deepflash2
from deepflash2.models import unet_falk2019 as _unet_falk2019
from deepflash2.models import unet_ronneberger2015 as _unet_ronneberger2015


def unet_deepflash2(pretrained=False, dataset='wue1_cFOS', **kwargs):
    """
    U-Net model optimized for deepflash2
    pretrained (bool): kwargs, load pretrained weights into the model
    dataset (string): specifies the dataset for pretrained weights (only applies if pretrained=True) 
    """
    model = _unet_deepflash2(pretrained=pretrained, dataset='wue1_cFOS', **kwargs)
    return model

def unet_falk2019(pretrained=False, dataset='wue1_cFOS', **kwargs):
    """
    U-Net model according to Falk, T., D. Mai, R. Bensch, Ö. Çiçek, A. Abdulkadir, Y. Marrakchi, … others. (2019). “U-Net: deep learning for cell counting, detection, and morphometry.” Nature Methods, 16 (1), 67–70.
    pretrained (bool): kwargs, load pretrained weights into the model
    dataset (string): specifies the dataset for pretrained weights (only applies if pretrained=True) 
    """
    model = _unet_falk2019(pretrained=pretrained, dataset='wue1_cFOS', **kwargs)
    return model

def unet_ronneberger2015(pretrained=False, dataset='wue1_cFOS', **kwargs):
    """
    Original U-Net model according to Ronneberger, O., P. Fischer and T. Brox. (2015). “U-net: Convolutional networks for biomedical image segmentation.” In: International Conference on Medical image computing and computer-assisted intervention (pp. 234–241).
    pretrained (bool): kwargs, load pretrained weights into the model
    dataset (string): specifies the dataset for pretrained weights (only applies if pretrained=True) 
    """
    model = _unet_ronneberger2015(pretrained=pretrained, dataset='wue1_cFOS', **kwargs)
    return model
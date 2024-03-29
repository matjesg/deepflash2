# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_config.ipynb (unless otherwise specified).

__all__ = ['Config']

# Cell
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch

# Cell
@dataclass
class Config:
    "Config class for settings."

    # Project
    project_dir:str = '.'

    # GT Estimation Settings
    # staple_thres:float = 0.5
    # staple_fval:int= 1
    vote_undec:int = 0

    # Train General Settings
    n_models:int = 5
    max_splits:int=5
    random_state:int = 42
    use_gpu:bool = True

    # Pytorch Segmentation Model Settings
    arch:str = 'Unet'
    encoder_name:str = 'tu-convnext_tiny'
    encoder_weights:str = 'imagenet'

    # Train Data Settings
    num_classes:int = 2
    tile_shape:int = 512
    scale:float = 1.
    instance_labels:bool = False

    # Train Settings
    base_lr:float = 0.001
    batch_size:int = 4
    weight_decay:float = 0.001
    mixed_precision_training:bool = True
    optim:str = 'Adam'
    loss:str = 'CrossEntropyDiceLoss'
    n_epochs:int = 25
    sample_mult:int = 0

    # Train Data Augmentation
    gamma_limit_lower:int = 80
    gamma_limit_upper:int = 120
    CLAHE_clip_limit:float = 0.0
    brightness_limit:float = 0.0
    contrast_limit:float = 0.0
    flip:bool = True
    rot:int = 360
    distort_limit:float = 0

    # Loss Settings
    mode:str = 'multiclass' #currently only tested for multiclass
    loss_alpha:float = 0.5 # Twerksky/Focal loss
    loss_beta:float = 0.5 # Twerksy Loss
    loss_gamma:float = 2.0 # Focal loss
    loss_smooth_factor:float = 0. #SoftCrossEntropyLoss

    # Pred/Val Settings
    use_tta:bool = True
    max_tile_shift: float = 0.5
    border_padding_factor:float = 0.25
    use_gaussian: bool = True
    gaussian_kernel_sigma_scale: float = 0.125
    min_pixel_export:int = 0

    # Instance Segmentation Settings
    cellpose_model:str='nuclei'
    cellpose_diameter:int=0
    cellpose_export_class:int=1
    cellpose_flow_threshold:float=0.4
    instance_segmentation_metrics:bool=False

    # Folder Structure
    gt_dir:str = 'GT_Estimation'
    train_dir:str = 'Training'
    pred_dir:str = 'Prediction'
    ens_dir:str = 'models'
    val_dir:str = 'valid'

    def __post_init__(self):
        self.set_device()

    def set_device(self, device:str=None):
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    @property
    def albumentation_kwargs(self):
        kwargs = ['gamma_limit_lower', 'gamma_limit_upper', 'CLAHE_clip_limit',
                  'brightness_limit', 'contrast_limit', 'distort_limit']
        return dict(filter(lambda x: x[0] in kwargs, self.__dict__.items()))

    @property
    def inference_kwargs(self):
        inference_kwargs = ['use_tta', 'max_tile_shift', 'use_gaussian', 'scale',
                            'gaussian_kernel_sigma_scale', 'border_padding_factor']
        return dict(filter(lambda x: x[0] in inference_kwargs, self.__dict__.items()))

    def save(self, path):
        'Save configuration to path'
        path = Path(path).with_suffix('.json')
        with open(path, 'w') as config_file:
            json.dump(asdict(self), config_file)
        print(f'Saved current configuration to {path}.json')
        return path

    def load(self, path):
        'Load configuration from path'
        path = Path(path)
        try:
            with open(path) as config_file: c = json.load(config_file)
            if not Path(c['project_dir']).is_dir(): c['project_dir']='deepflash2'
            for k,v in c.items(): setattr(self, k, v)
            print(f'Successsfully loaded configuration from {path}')
        except:
            print('Error! Select valid config file (.json)')
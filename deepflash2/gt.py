# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/09_gt.ipynb (unless otherwise specified).

__all__ = ['import_sitk', 'staple_multi_label', 'm_voting', 'GTEstimator']

# Cell
import imageio, pandas as pd, numpy as np
from pathlib import Path
from fastcore.basics import GetAttr
from fastprogress import progress_bar
from fastai.data.transforms import get_image_files

import matplotlib.pyplot as plt

from .data import _read_msk
from .config import Config
from .utils import clean_show, save_mask, dice_score, install_package, get_instance_segmentation_metrics

# Cell
def import_sitk():
    try:
        import SimpleITK
        assert SimpleITK.Version_MajorVersion()==2
    except:
        print('Installing SimpleITK. Please wait.')
        install_package("SimpleITK==2.1.1.1")
    import SimpleITK
    return SimpleITK

# Cell
def staple_multi_label(segmentations, label_undecided_pixel=1):
    'STAPLE: Simultaneous Truth and Performance Level Estimation with simple ITK'
    sitk = import_sitk()
    sitk_segmentations = [sitk.GetImageFromArray(x) for x in segmentations]
    STAPLE = sitk.MultiLabelSTAPLEImageFilter()
    STAPLE.SetLabelForUndecidedPixels(label_undecided_pixel)
    msk = STAPLE.Execute(sitk_segmentations)
    return sitk.GetArrayFromImage(msk)

# Cell
def m_voting(segmentations, labelForUndecidedPixels = 0):
    'Majority Voting from  simple ITK Label Voting'
    sitk = import_sitk()
    segmentations = [sitk.GetImageFromArray(x) for x in segmentations]
    mv_segmentation = sitk.LabelVoting(segmentations, labelForUndecidedPixels)
    return sitk.GetArrayFromImage(mv_segmentation)

# Cell
class GTEstimator(GetAttr):
    "Class for ground truth estimation"
    _default = 'config'

    def __init__(self, exp_dir='expert_segmentations', config=None, path=None, cmap='viridis' , verbose=1):
        self.exp_dir = exp_dir
        self.config = config or Config()
        self.path = Path(path) if path is not None else Path('.')
        self.mask_fn = lambda exp,msk: self.path/self.exp_dir/exp/msk
        self.cmap = cmap
        self.gt = {}

        f_list = get_image_files(self.path/self.exp_dir)
        assert len(f_list)>0, f'Found {len(f_list)} masks in "{self.path/self.exp_dir}". Please check your masks and expert folders.'
        ass_str = f'Found unexpected folder structure in {self.path/self.exp_dir}. Please check your provided masks and folders.'
        assert len(f_list[0].relative_to(self.path/self.exp_dir).parents)==2, ass_str

        self.masks = {}
        self.experts = []
        for m in sorted(f_list):
            exp = m.parent.name
            if m.name in self.masks:
                self.masks[m.name].append(exp)
            else:
                self.masks[m.name] = [exp]
            self.experts.append(exp)
        self.experts = sorted(set(self.experts))
        if verbose>0: print(f'Found {len(self.masks)} unique segmentation mask(s) from {len(self.experts)} expert(s)')

    def show_data(self, max_n=6, files=None, figsize=None, **kwargs):
        if files is not None:
            files = [(m,self.masks[m]) for m in files]
        else:
            max_n = min((max_n, len(self.masks)))
            files = list(self.masks.items())[:max_n]
        if not figsize: figsize = (len(self.experts)*3,3)
        for m, exps in files:
            fig, axs = plt.subplots(nrows=1, ncols=len(exps), figsize=figsize, **kwargs)
            vkwargs = {'vmin':0, 'vmax':self.num_classes-1}
            for i, exp in enumerate(exps):
                msk = _read_msk(self.mask_fn(exp,m), num_classes=self.num_classes, instance_labels=self.instance_labels)
                if i == len(exps) - 1:
                    clean_show(axs[i], msk, exp, self.cmap, cbar='classes', ticks=self.num_classes, **vkwargs)
                else:
                    clean_show(axs[i], msk, exp, self.cmap, **vkwargs)
            fig.text(0, .5, m, ha='center', va='center', rotation=90)
            plt.tight_layout()
            plt.show()

    def gt_estimation(self, method='STAPLE', save_dir=None, filetype='.png', **kwargs):
        assert method in ['STAPLE', 'majority_voting']
        res = []
        refs = {}
        print(f'Starting ground truth estimation - {method}')
        for m, exps in progress_bar(self.masks.items()):
            masks = [_read_msk(self.mask_fn(exp,m), num_classes=self.num_classes, instance_labels=self.instance_labels) for exp in exps]
            if method=='STAPLE':
                #ref = staple(masks, self.staple_fval, self.staple_thres)
                ref = staple_multi_label(masks, self.vote_undec)
            elif method=='majority_voting':
                ref = m_voting(masks, self.vote_undec)
            refs[m] = ref
            #assert ref.mean() > 0, 'Please try again!'
            df_tmp = pd.DataFrame({'method': method, 'file' : m, 'exp' : exps, 'dice_score': [dice_score(ref, msk, num_classes=self.num_classes) for msk in masks]})
            if self.instance_segmentation_metrics:
                mAP, AP = [],[]
                for msk in masks:
                    ap, tp, fp, fn = get_instance_segmentation_metrics(ref, msk, is_binary=True, **kwargs)
                    mAP.append(ap.mean())
                    AP.append(ap[0])
                df_tmp['mean_average_precision'] = mAP
                df_tmp['average_precision_at_iou_50'] = AP
            res.append(df_tmp)
            if save_dir:
                path = self.path/save_dir
                path.mkdir(exist_ok=True, parents=True)
                save_mask(ref, path/Path(m).stem, filetype)
        self.gt[method] = refs
        self.df_res = pd.concat(res)
        self.df_agg = self.df_res.groupby('exp').agg(average_dice_score=('dice_score', 'mean'), std_dice_score=('dice_score', 'std'))
        if self.instance_segmentation_metrics:
            self.df_agg = self.df_res.groupby('exp').agg(average_dice_score=('dice_score', 'mean'),
                                                         std_dice_score=('dice_score', 'std'),
                                                         average_mean_average_precision=('mean_average_precision', 'mean'),
                                                         std_mean_average_precision=('mean_average_precision', 'std'),
                                                         average_average_precision_at_iou_50=('average_precision_at_iou_50', 'mean'),
                                                         std_average_precision_at_iou_50=('average_precision_at_iou_50', 'std'))
        if save_dir:
            self.df_res.to_csv(path.parent/f'{method}_vs_experts.csv', index=False)
            self.df_agg.to_csv(path.parent/f'{method}_vs_experts_agg.csv', index=False)
            with pd.ExcelWriter(path.parent/f'{method}_vs_experts.xlsx') as writer:
                self.df_res.to_excel(writer, sheet_name='raw')
                self.df_agg.to_excel(writer, sheet_name='aggregated')

    def show_gt(self, method='STAPLE', max_n=6, files=None, figsize=(15,7), **kwargs):
        from IPython.display import Markdown, display
        if not files: files = list(self.masks.keys())[:max_n]
        for f in files:
            if self.num_classes==2:
                fig, ax = plt.subplots(ncols=2, figsize=figsize, **kwargs)
                # GT
                clean_show(ax[0], self.gt[method][f], f'{method} (binary mask)', cbar='', cmap=self.cmap)
                # Experts
                masks = [_read_msk(self.mask_fn(exp,f), num_classes=self.num_classes, instance_labels=self.instance_labels) for exp in self.masks[f]]
                masks_av = np.array(masks).sum(axis=0)#/len(masks)
                clean_show(ax[1], masks_av, 'Expert Overlay', cbar='experts', ticks=len(masks), cmap=plt.cm.get_cmap(self.cmap, len(masks)+1))
            else:
                fig, ax = plt.subplots(ncols=1, figsize=figsize, **kwargs)
                clean_show(ax, self.gt[method][f], f'{method}', cbar='classes', cmap=self.cmap, ticks=self.num_classes)
            # Results
            metrics = ['dice_score', 'mean_average_precision', 'average_precision_at_iou_50'] if self.instance_segmentation_metrics else ['dice_score']
            av_df = pd.DataFrame([self.df_res[self.df_res.file==f][metrics].mean()], index=['average'], columns=metrics)
            plt_df = self.df_res[self.df_res.file==f].set_index('exp')[metrics].append(av_df)
            fig.text(0, .5, f, ha='center', va='center', rotation=90)
            plt.tight_layout()
            plt.show()
            display(plt_df)
            display(Markdown('---'))

# Reproducing the results of our paper

- **Paper**: The Nature Communications article is available at [here](https://www.nature.com/articles/s41467-023-36960-9).
- **Data**: Data is available on [Google Drive](https://drive.google.com/drive/folders/1r9AqP9qW9JThbMIvT0jhoA5mPxWEeIjs?usp=sharing). 
- **Models**: Trained models are available in the model libary on [github](https://github.com/matjesg/deepflash2/releases/tag/model_library). 
- **Benchmark Models**: Trained benchmark models are available on [Google Drive](https://drive.google.com/drive/folders/1BZRrRTDuJw5EoBqz1RWoFKZ7eq2kEwxm?usp=sharing). 

## deepflash2 notebooks

Notebooks reproduce the results of our paper:

1. Ground truth estimation [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/1_gt_estimation.ipynb) 

2. Ensemble training and prediction [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/2_train_and_predict.ipynb) 

3. Performance comparison between deepflash2, experts, and benchmark methods [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/3_performance_comparison.ipynb) 
  - 3.1 Performance comparion for the reliabiliy of results [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/3-1_performance_comparison_reliability.ipynb) 
  - 3.2 Performance comparion for challenge data [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/3-2_performance_challenge_data.ipynb) 

4. Relationship between uncertainty and expert agreement [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/4_experts_vs_uncertainties.ipynb) 

5. Out-of-distribution detection [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/5_ood_detection.ipynb) 

6. Figures and Tables [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/6_figures_tables.ipynb) 

The figure and table source data is also available [here](https://github.com/matjesg/deepflash2/releases/tag/paper_source_data).


## Benchmark methods

Notebooks reproduce the benchmark methods in our paper (including model training).

- [U-Net](https://lmb.informatik.uni-freiburg.de/resources/opensource/unet/) (semantic segmentation and instance segmentation) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/benchmark_unet_2019.ipynb) 
  - Falk, T., Mai, D., Bensch, R., Çiçek, Ö., Abdulkadir, A., Marrakchi, Y., ... & Ronneberger, O. (2019). U-Net: deep learning for cell counting, detection, and morphometry. Nature methods, 16(1), 67-70.
- [nnunet](https://github.com/MIC-DKFZ/nnUNet) (semantic segmentation and instance segmentation) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/benchmark_nnunet.ipynb) 
  - Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
- [cellpose](https://github.com/MouseLand/cellpose) (instance segmentation) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/benchmark_cellpose.ipynb) 
  - Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods, 18(1), 100-106.
- Otsu Thresholding (semantic segmentation) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matjesg/deepflash2/blob/master/paper/benchmark_otsu.ipynb) 
  - Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics, 9(1), 62-66.

The trained models are available on [Google Drive](https://drive.google.com/drive/folders/1BZRrRTDuJw5EoBqz1RWoFKZ7eq2kEwxm?usp=sharing).


## Using the data in Google Drive with Colab

To use the data in Google Colab, create a [shortcut](https://support.google.com/drive/answer/9700156) of the data folder in your personal Google Drive. This will not use your Google storage space.

![image](https://user-images.githubusercontent.com/13711052/144764480-8efc4189-4df3-499d-aac4-dd20efe8d86b.png)

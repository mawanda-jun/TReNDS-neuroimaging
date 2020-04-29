# Articles
## Lists of useful articles
[Brain-Image-Analysis](https://github.com/largeapp/Brain-Image-Analysis#sMRI-and-other-data)

## sMRI SBC
[Source-Based Morphometry: The Use of Independent Component Analysis to Identify Gray Matter Differences With Application to Schizophrenia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2751641/) 
Independent component analysis (ICA) is a popular statistical and computational technique for biomedical signal analysis. The biomedical signals that we can measure are often mixtures of signals from different underlying “sources,” including both noise or signals of interest. ICA works by decomposing the mixed signals into maximally independent components. ICA has shown considerable promise for the analysis of fMRI [Calhoun and Adali, 2006] and EEG data [Makeig et al., 1997] and also for segmenting the gray matter and white matter in sMRI [Nakai et al., 2004]. Here we propose the use of ICA to extract maximally spatially independent sources revealing patterns of variation that occur in sMRI images and to identify sMRI differences between patients and healthy controls. We hypothesized that a small number of sources in the brain would show differences between patients and healthy controls. Under this assumption, we can apply ICA to the preprocessed sMRI images, identify the sources, and perform statistical analysis to identify which sources distinguish patients from healthy controls. We refer to this straightforward but effective approach as source-based morphometry 
(SBM).  

## FNC 

## Architectures and pre-processing
### Useful for pre-processing
- [Modeling 4D fMRI Data via Spatio-Temporal Convolutional Neural Networks (ST-CNN)](https://arxiv.org/pdf/1805.12564.pdf)
**Abstract**. Simultaneous modeling of the spatio-temporal variation patterns of brain functional network from 4D fMRI data has been an important yet challenging problem for the field of cognitive neuroscience
and medical image analysis. Inspired by the recent success in applying
deep learning for functional brain decoding and encoding, in this work
we propose a spatio-temporal convolutional neural network (ST-CNN)
to jointly learn the spatial and temporal patterns of targeted network
from the training data and perform automatic, pin-pointing functional
network identification. The proposed ST-CNN is evaluated by the task
of identifying the Default Mode Network (DMN) from fMRI data. Results show that while the framework is only trained on one fMRI dataset,
it has the sufficient generalizability to identify the DMN from different
populations of data as well as different cognitive tasks. Further investigation into the results show that the superior performance of ST-CNN
is driven by the jointly-learning scheme, which capture the intrinsic relationship between the spatial and temporal characteristic of DMN and
ensures the accurate identification.
- [Machine Learning for Neuroimaging with Scikit-Learn](https://arxiv.org/pdf/1412.3919.pdf)

### Useful for the implementation
- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) e [link GitHub](https://github.com/StanfordVL/MinkowskiEngine)





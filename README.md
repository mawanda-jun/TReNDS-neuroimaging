Implementation for the [TReNDS neuroimaging Kaggle Competition](https://www.kaggle.com/c/trends-assessment-prediction/).

# Repository state
I decided to concentrate on the images dataset. 
Unfortunately, they are not well suited to gain very high scores - not without the tabular data.
This is why I decided to move on to another competition, where the visual inputs are effectively used to understand the problem.
For this reason, this repository's state is stopped. 

However, I was glad to learn many concept with which I was unfamiliar with and I'm going to explain the main gains I obtained.

# Dataset simplification
The whole data is made up of ~460GB of 3D [fMRI ICA scans](https://www.kaggle.com/c/trends-assessment-prediction/discussion/155833).
This data is therefore composed of 5.877 train and 5.877 test volumes of dimension `(53,52,63,53)` - spatial dimension `(52,63,53)`.
Each volume is stored inside a `h5py` dataset, `float64` datatype, which is a greater limitation with parallel reading from disk:
I decided to convert the whole dataset to a `float32` `torch.Tensor`. 
In this way I was able to limit the overall dimension (~500GB) and to use the `torch.load` method - which is way more faster then the `h5py` one.
In addition, I normalized and standardized the dataset with norm and variance, so there is no need to calculate it online.
I also tried other approaches - as discussed [here](https://www.kaggle.com/c/trends-assessment-prediction/discussion/148864) - but they were less efficient.

I managed the dataset with the file `manage_dataset.py`.

# Very fast loading and transformations with PyTorch Dataloader API and MONAI
To load the entire dataset I used the `Dataloader` PyTorch API, which let me load and transform the dataset in a very fast and customizable way.

The dataset is customized so only the strictly necessary data is passed to the model at a time - keeping the change to load very different type of data with a single flag.
I tried many different approaches with the [MONAI](https://monai.io/) library, which let me modify the images without effort:
- translations;
- pixel shifting and scaling;
- random rotating;
- Gaussian noise;
- resizing;
- cropping.

 
I found that the CPU was severely used in these processes, so the use of a secondary GPU would have been better in order to gain speed.

The approaches are easily accessible in the file `dataset.py`.

# Networks
I followed and trained three different network flavours: straight CNN, CNN with siamese, and their sparse variation. 
In addition, I tried a [classification with a VAE as regularization term](https://github.com/mawanda-jun/3D-mri-brain-features-classification-using-autoencoder-regularization), but I didn't invest much time in its training since it was a complex approach to the problem.

## CNN regression
I used various complexity of the ResNet architecture to train my model, taking as "features channel" the dimension of the independent components stacked on the first dimension of the tensor `(53)`.
This approach led me to the most promising results - `0.714` on LB score -, with a ResNet10 3D.

## Siamese CNN regression
In this approach, I interpreted the independent components as different images, from which the network would be able to understand the differences - and the correlations - between them.
This approach is relatively more difficult to train and very GPU memory demanding. The main bottleneck was the GPU memory, which is filled even with small CNN networks.

## Sparse CNN
The fMRI ICA images are relatively sparse - with a threshold of (-3,3), they are ~95% zeros - so I decided to implement the [FacebookResearch SparseConvNet library](https://github.com/facebookresearch/SparseConvNet).
Unfortunately I definitely didn't find any advantage over the use of its dense representation. 
This experiment has been really helpful in the understanding of the sparse representation - and in how to deal with color channels and batch size while using those kind of libraries.

The custom `collate_fn` to produce the right data are attached to the network class inside the `SparseResNet.py` file, together with the networks implementations.

# Hyperparameters search
To understand how to find the optimal learning rate - given a set of other hyperparameters - I decided to follow the indication of the paper regarding the [cyclic learning rates for neural networks](https://arxiv.org/abs/1506.01186).
This research gave me an important boost in the development of new architectures - and to rapidly understand the learning capacity of each one. 
In particular, I made use of the library [torch-lr-finder](https://pypi.org/project/torch-lr-finder/), which is a easy-to-learn tool to apply the method described in the paper.

# Half precision training
I decided to use the half precision for my models in order to guarantee higher batch sizes. 
Therefore, I used [apex](https://github.com/NVIDIA/apex) which is a tools for easy mixed precision and distributed training in Pytorch. More on the attached link.

# Deployment
In order to test and train with those experimental frameworks, I decided to embrace the docker philosophy and I used the [NVidia-docker](https://github.com/NVIDIA/nvidia-docker) with the help of the [PyTorch image](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch).

# Machine setup
I tested the above configuration on this machine:
- CPU: AMD 3600X
- GPU: NVidia 2070s
- SSD: 1TB Samsung 970 evo plus
- RAM: 64GB GSkill 3200MHz

 






 


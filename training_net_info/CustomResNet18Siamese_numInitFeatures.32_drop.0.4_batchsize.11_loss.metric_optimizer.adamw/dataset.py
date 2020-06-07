import os
from h5py import File as h5File
from tqdm import tqdm
import numpy as np
import pandas as pd
import zlib
from torch.utils.data import Dataset
import torch
from monai.transforms import \
    RandAffine, \
    RandGaussianNoise, \
    RandShiftIntensity, \
    RandSpatialCrop, \
    Resize, \
    RandFlip, \
    RandScaleIntensity, \
    RandRotate


class TReNDS_dataset(Dataset):
    def __init__(self, pt_folder, sbm_path, train_scores_path=None, fnc_path=None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .pt file as a dictionary {patientID: complete_path_to_file}
        self.pt_paths = {int(filename.split('.')[0]): os.path.join(pt_folder, filename) for filename in
                         os.listdir(pt_folder)}

        if fnc_path:
            fnc = pd.read_csv(
                fnc_path)  # There are no NaN values here (ref: https://www.kaggle.com/rftexas/trends-in-dept-understanding-eda-lgb-baseline)
            self.fnc = {Id: np.array(fnc.loc[fnc['Id'] == Id]).squeeze()[1:] for Id in self.pt_paths.keys()}
        else:
            self.fnc = None

        sbm = pd.read_csv(sbm_path)  # There are no NaN values here (as before)
        self.sbm = {Id: np.array(sbm.loc[sbm['Id'] == Id]).squeeze()[1:] for Id in self.pt_paths.keys()}

        # ICN_num = pd.read_csv(ICN_numbers_path)  # There are no NaN values
        # self.ICN_num = np.array(ICN_num['ICN_number']).squeeze()

        # Check if dataset is for training or for submission
        if train_scores_path:
            train_scores = pd.read_csv(train_scores_path)
            train_scores.fillna(train_scores.mean(),
                                inplace=True)  # Look for NaN values and replace them with column mean
            self.labels = {Id: np.array(train_scores.loc[train_scores['Id'] == Id]).squeeze()[1:] for Id in
                           self.pt_paths.keys()}
        else:
            self.labels = None

        # Test code to verify if there are all the labels for each type of data
        # fnc_keys = list(fnc['Id'])
        # sbm_keys = list(sbm['Id'])
        # print(len(pt_keys), len(fnc_keys), len(sbm_keys))
        # fnc_missing = []
        # sbm_missing = []
        # for k in pt_keys:
        #     if k not in fnc_keys:
        #         fnc_missing.append(k)
        #     if k not in sbm_keys:
        #         sbm_missing.append(k)
        # print(fnc_missing, sbm_missing)
        # pass

        # self.mask = np.array(nib.load(mask_path))

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.pt_paths.keys())}
        # Create reverse order to have control over dataset patients IDs and indexes
        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.pt_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        ID = self.__num_to_id[item]

        # Logic to put items in new sample - since not every item is read during training:
        # ID: 1
        # sbm: 2
        # brain: 3
        # fnc: 4 - it may be not read during training
        # label: bottom - during test there is no label to be read
        sample = ID, self.sbm[ID], torch.load(self.pt_paths[ID]).numpy()

        # Add fnc if needed
        if self.fnc:
            sample = *sample, self.fnc[ID]

        # Add labels to the sample if the dataset is the training one.
        if self.labels:
            sample = *sample, self.labels[ID]

        # Transform sample (if defined)
        self.previous = sample
        return self.transform(sample) if self.transform is not None else sample


# Custom transform example
class ToTensor:
    def __init__(self, use_fnc=False, train=True, lr_range=False):
        self.use_fnc = use_fnc
        self.train = train
        self.lr_range = lr_range

    def __call__(self, sample):
        # ID = sample[0]
        # sbm = sample[1]
        # brain = sample[2]
        # fnc ?= sample[3]
        # labels ?= sample[-1]
        new_sample = sample[0], torch.tensor(sample[1], dtype=torch.float32)

        brain = sample[2]
        if brain is not None:
            # Define use of brain images - which are not necessary for shallow networks
            # Brain is a cpu tensor already, so no need for transformations
            new_sample = *new_sample, torch.tensor(brain, dtype=torch.float32)

        if self.use_fnc:
            new_sample = *new_sample, torch.tensor(sample[3], dtype=torch.float32)
        if self.train:  # Add label and skip ID
            new_sample = *new_sample[1:], torch.tensor(sample[-1], dtype=torch.float32)

        # Return inputs and labels, separately.
        return new_sample[:-1], new_sample[-1]


class Normalize:
    """
    Normalize brain images with mean/std calculated with Welford's algorithm
    """

    def __init__(self, mean_path, std_path):
        self.mean = torch.load(mean_path).numpy()
        self.std = torch.load(std_path).numpy()
        self.std[self.std == 0] = 100

    def __call__(self, sample, *args, **kwargs):
        brain: np.ndarray = sample[2]
        brain = (brain - self.mean) / self.std
        # Mean and variance are in float64 precision, so we need to cast `brain` to float32 again.
        sample = *sample[0:2], brain.astype('float32'), *sample[3:]
        return sample


class fMRI_Aumentation:
    def __init__(self):
        self.rand_affine = RandAffine(
            # mode='nearest',
            prob=0.5,
            # spatial_size=(52, 63, 53),  # Original spatial size
            translate_range=(5, 5, 5),
            rotate_range=(np.pi * 4, np.pi * 4, np.pi * 4),
            # scale_range=(0.15, 0.15, 0.15),
            # padding_mode='zeros',
            as_tensor_output=False
        )
        # self.gaussian_noise = RandGaussianNoise(prob=.5)
        self.rand_scale_intensity = RandScaleIntensity(1., prob=1.)
        self.rand_shift_intensity = RandShiftIntensity(0.1, prob=1.)
        # self.rand_flip = RandFlip(spatial_axis=(0, 1, 2), prob=0.5)  # The axis is 0, 1, 2 are without colors channel
        # self.crop = RandSpatialCrop(roi_size=(35, 35, 35), random_center=True, random_size=True)
        # self.resize = Resize((52, 63, 53), mode='wrap')

    def __call__(self, sample, *args, **kwargs):
        brain: np.ndarray = sample[2]
        brain = self.rand_affine(brain, (52, 63, 53))
        brain = self.rand_scale_intensity(brain)
        brain = self.rand_shift_intensity(brain)
        sample = *sample[0:2], brain, *sample[3:]
        return sample


class RandomCropToDim:
    def __init__(self):
        self.random_crop = RandSpatialCrop(roi_size=(48, 48, 48), random_size=False)

    def __call__(self, sample, *args, **kwargs):
        brain: np.ndarray = sample[2]
        brain = self.random_crop(brain)
        sample = *sample[0:2], brain, *sample[3:]
        return sample


class ResizeToDim:
    def __init__(self):
        self.resize = Resize((48, 48, 48))

    def __call__(self, sample, *args, **kwargs):
        brain: np.ndarray = sample[2]
        brain = self.resize(brain)
        sample = *sample[0:2], brain, *sample[3:]
        return sample


class AugmentDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, aug_fn):
        self.dataset = dataset
        self.aug_fn = aug_fn

    def __getitem__(self, index):
        return self.aug_fn(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    base_path = '/opt/dataset/'
    train_pt_folder = os.path.join(base_path, 'fMRI_train_norm')
    test_pt_folder = os.path.join(base_path, 'fMRI_test_torch')
    fnc_path = os.path.join(base_path, 'Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'Kaggle/fMRI_mask.nii')
    mean_path = os.path.join(base_path, 'mean.pt')
    variance_path = os.path.join(base_path, 'variance.pt')

    # Define transformations
    trans = transforms.Compose([ToTensor()])

    dataset = TReNDS_dataset(train_pt_folder, sbm_path, None, None, transform=trans)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, pin_memory=True, num_workers=12)

    for batch in tqdm(dataloader, desc='Reading dataset...'):
        pass

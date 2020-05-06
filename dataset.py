import os
from h5py import File as h5File
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
import zlib
from torch.utils.data import Dataset
import torch
from monai.transforms import RandAffine


class TReNDS_dataset(Dataset):
    def __init__(self, pt_folder, sbm_path, train_scores_path=None, fnc_path=None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .pt file as a dictionary {patientID: complete_path_to_file}
        self.pt_paths = {int(filename.split('.')[0]): os.path.join(pt_folder, filename) for filename in os.listdir(pt_folder)}

        if fnc_path:
            fnc = pd.read_csv(fnc_path)  # There are no NaN values here (ref: https://www.kaggle.com/rftexas/trends-in-dept-understanding-eda-lgb-baseline)
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
            train_scores.fillna(train_scores.mean(), inplace=True)  # Look for NaN values and replace them with column mean
            self.labels = {Id: np.array(train_scores.loc[train_scores['Id'] == Id]).squeeze()[1:] for Id in self.pt_paths.keys()}
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

        # Retrieve all information from the Dataset initialization
        # Keep brain commented until not working on 3D images
        brain = torch.load(self.pt_paths[ID])
        # brain = np.copy(np.frombuffer(zlib.decompress(open(self.pt_paths[ID], 'rb').read()), dtype='float64').reshape(53, 52, 63, 53))
        # brain = None

        sbm = self.sbm[ID]
        # Create sample
        sample = {
            'ID': ID,
            'sbm': sbm,
            'brain': brain
        }
        if self.fnc:
            sample['fnc'] = self.fnc[ID]
        # Add labels to the sample if the dataset is the training one.
        if self.labels:
            sample['label'] = self.labels[ID]

        # Transform sample (if defined)
        return self.transform(sample) if self.transform is not None else sample


# Custom transform example
class ToTensor:
    def __call__(self, sample):
        sbm = torch.tensor(sample['sbm']).float()
        ID = sample['ID']

        new_sample = {'sbm': sbm, 'ID': ID}

        if sample['brain'] is not None:
            # Define use of brain images - which are not necessary for shallow networks
            # Brain is a cpu tensor already, so no need for transformations
            new_sample['brain'] = sample['brain']
        # Define list of keys in sample to avoid the use of try/except which is quite computational costy
        sample_keys = list(sample.keys())
        tries_keys = ['fnc', 'label']
        for tk in tries_keys:
            if tk in sample_keys:
                new_sample[tk] = torch.tensor(sample[tk]).float()

        return new_sample


class Normalize:
    """
    Normalize brain images with mean/std calculated with Welford's algorithm
    """
    def __init__(self, mean_path, std_path):
        self.mean = torch.load(mean_path)
        self.std = torch.load(std_path)
        self.std[self.std == 0] = 100

    def __call__(self, sample, *args, **kwargs):
        brain = sample['brain']
        brain = (brain - self.mean) / self.std
        # Mean and variance are in float64 precision, so we need to cast `brain` to float32 again.
        return {**sample, 'brain': brain.float()}


class fMRI_Aumentation:
    def __init__(self):
        self.rand_affine = RandAffine(
            mode='nearest',
            prob=0.5,
            spatial_size=(52, 63, 53),
            translate_range=(5, 5, 5),
            rotate_range=(np.pi*4, np.pi*4, np.pi*4),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode='border',
            # device=torch.device('cuda:0')
        )

    def __call__(self, sample, *args, **kwargs):
        brain: np.ndarray = sample['brain']  #.to('cuda:0)
        brain = self.rand_affine(brain, (52, 63, 53))
        return {**sample, 'brain': brain}  #.cpu()


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

    base_path = '..'
    train_pt_folder = os.path.join(base_path, 'dataset/fMRI_train_torch')
    test_pt_folder = os.path.join(base_path, 'dataset/fMRI_test_torch')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')
    mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    trans = transforms.Compose([ToTensor(), fMRI_Aumentation()])

    dataset = TReNDS_dataset(train_pt_folder, sbm_path, None, None, transform=trans)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, pin_memory=True, num_workers=8)

    for batch in tqdm(dataloader, desc='Reading dataset...'):
        pass




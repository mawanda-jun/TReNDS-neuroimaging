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
    def __init__(self, gz_folder, sbm_path, fnc_path, train_scores_path = None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .gz file as a dictionary {patientID: complete_path_to_file}
        self.gz_paths = {int(filename.split('.')[0]): os.path.join(gz_folder, filename) for filename in os.listdir(gz_folder)}

        if fnc_path:
            fnc = pd.read_csv(fnc_path)  # There are no NaN values here (ref: https://www.kaggle.com/rftexas/trends-in-degz-understanding-eda-lgb-baseline)
            self.fnc = {Id: np.array(fnc.loc[fnc['Id'] == Id]).squeeze()[1:] for Id in self.gz_paths.keys()}
        else:
            self.fnc = None

        sbm = pd.read_csv(sbm_path)  # There are no NaN values here (as before)
        self.sbm = {Id: np.array(sbm.loc[sbm['Id'] == Id]).squeeze()[1:] for Id in self.gz_paths.keys()}

        # ICN_num = pd.read_csv(ICN_numbers_path)  # There are no NaN values
        # self.ICN_num = np.array(ICN_num['ICN_number']).squeeze()

        # Check if dataset is for training or for submission
        if train_scores_path:
            train_scores = pd.read_csv(train_scores_path)
            train_scores.fillna(train_scores.mean(), inplace=True)  # Look for NaN values and replace them with column mean
            self.labels = {Id: np.array(train_scores.loc[train_scores['Id'] == Id]).squeeze()[1:] for Id in self.gz_paths.keys()}
        else:
            self.labels = None

        # Test code to verify if there are all the labels for each type of data
        # fnc_keys = list(fnc['Id'])
        # sbm_keys = list(sbm['Id'])
        # print(len(gz_keys), len(fnc_keys), len(sbm_keys))
        # fnc_missing = []
        # sbm_missing = []
        # for k in gz_keys:
        #     if k not in fnc_keys:
        #         fnc_missing.append(k)
        #     if k not in sbm_keys:
        #         sbm_missing.append(k)
        # print(fnc_missing, sbm_missing)
        # pass

        # self.mask = np.array(nib.load(mask_path))

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.gz_paths.keys())}
        # Create reverse order to have control over dataset patients IDs and indexes
        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.gz_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        ID = self.__num_to_id[item]

        # Retrieve all information from the Dataset initialization
        # Keep brain commented until not working on 3D images
        # brain = torch.load(self.gz_paths[ID])
        brain = np.frombuffer(zlib.decompress(open(self.gz_paths[ID], 'rb').read()), dtype='float64').reshape(53, 52, 63, 53)
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
        return self.transform(sample) if self.transform else sample


# Custom transform example
class ToTensor:
    def __call__(self, sample):
        fnc = torch.tensor(sample['fnc']).float()
        sbm = torch.tensor(sample['sbm']).float()
        ID = torch.tensor(sample['ID']).float()

        new_sample = {'fnc': fnc, 'sbm': sbm, 'ID': ID}

        if sample['brain'] is not None:
            # Define use of brain images - which are not necessary for shallow networks
            # If coming from torch tensors, they have already been summed and normalized
            new_sample['brain'] = torch.tensor(sample['brain']).float()
        try:
            # Look for fnc key. If not present - when using 3D images - there is no need to store these values
            new_sample['fnc'] = torch.tensor(sample['fnc']).float()
        except KeyError:
            pass

        try:
            # Look for label key. If not present, it is the test set, so no need to store the label.
            new_sample['label'] = torch.tensor(sample['label']).float()
        except KeyError:
            pass
        finally:
            return new_sample


class Normalize:
    """
    Normalize brain images with mean/std calculated with Welford's algorithm
    """
    def __init__(self, mean_path, std_path):
        self.mean = torch.load(mean_path).numpy()
        self.std = torch.load(std_path).numpy()

    def __call__(self, sample, *args, **kwargs):
        brain = sample['brain']
        brain = (brain - self.mean) / self.std
        return {**sample, 'brain': brain}


class fMRI_Aumentation:
    def __init__(self):
        self.rand_affine = RandAffine(
            mode=('bilinear', 'nearest'),
            prob=0.5,
            spatial_size=(52, 63, 53),
            translate_range=(5, 5, 5),
            rotate_range=(np.pi*4, np.pi*4, np.pi*4),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode='border'
        )

    def __call__(self, sample, *args, **kwargs):
        brain: np.ndarray = sample['brain']
        brain = self.rand_affine(brain, (52, 63, 53))
        return {**sample, 'brain': brain}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    base_path = '..'
    train_gz_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train_compressed')
    test_gz_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_test_torch')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')
    mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    trans = transforms.Compose([Normalize(mean_path, variance_path), fMRI_Aumentation(), transforms.ToTensor()])

    dataset = TReNDS_dataset(train_gz_folder, sbm_path, None, None, transform=trans)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)

    for batch in dataloader:
        print(batch['brain'].shape)




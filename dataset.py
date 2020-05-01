import os
from h5py import File as h5File
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import torch


class TReNDS_dataset(Dataset):
    def __init__(self, mat_folder, fnc_path, sbm_path, ICN_numbers_path, train_scores_path = None, mask_path = None, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .mat file as a dictionary {patientID: complete_path_to_file}
        self.mat_paths = {int(filename.split('.')[0]): os.path.join(mat_folder, filename) for filename in os.listdir(mat_folder)}

        fnc = pd.read_csv(fnc_path)  # There are no NaN values here (ref: https://www.kaggle.com/rftexas/trends-in-depth-understanding-eda-lgb-baseline)
        self.fnc = {Id: np.array(fnc.loc[fnc['Id'] == Id]).squeeze()[1:] for Id in self.mat_paths.keys()}

        sbm = pd.read_csv(sbm_path)  # There are no NaN values here (as before)
        self.sbm = {Id: np.array(sbm.loc[sbm['Id'] == Id]).squeeze()[1:] for Id in self.mat_paths.keys()}

        ICN_num = pd.read_csv(ICN_numbers_path)  # There are no NaN values
        self.ICN_num = np.array(ICN_num['ICN_number']).squeeze()

        train_scores = pd.read_csv(train_scores_path) if train_scores_path else None
        train_scores.fillna(train_scores.mean(), inplace=True)  # Look for NaN values and replace them with column mean
        self.labels = {Id: np.array(train_scores.loc[train_scores['Id'] == Id]).squeeze()[1:] for Id in self.mat_paths.keys()} if train_scores_path else None

        # Test code to verify if there are all the labels for each type of data
        # fnc_keys = list(fnc['Id'])
        # sbm_keys = list(sbm['Id'])
        # print(len(mat_keys), len(fnc_keys), len(sbm_keys))
        # fnc_missing = []
        # sbm_missing = []
        # for k in mat_keys:
        #     if k not in fnc_keys:
        #         fnc_missing.append(k)
        #     if k not in sbm_keys:
        #         sbm_missing.append(k)
        # print(fnc_missing, sbm_missing)
        # pass

        self.mask = np.array(nib.load(mask_path))

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.mat_paths.keys())}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.mat_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        id = self.__num_to_id[item]

        # Retrieve all information from the Dataset initialization
        # Keep brain commented until not working on 3D images
        # brain = np.array(h5File(self.mat_paths[id], 'r')['SM_feature'])
        brain = None
        fnc = self.fnc[id]
        sbm = self.sbm[id]
        # Create sample
        sample = {
            'fnc': fnc,
            'sbm': sbm,
            'brain': brain
        }

        # Add labels to the sample if the dataset is the training one.
        if self.labels:
            sample['label'] = self.labels[id]

        # Transform sample (if defined)
        return self.transform(sample) if self.transform else sample


# Custom transform example
class ToTensor:
    def __call__(self, sample):
        fnc = torch.tensor(sample['fnc']).float()
        sbm = torch.tensor(sample['sbm']).float()
        label = torch.tensor(sample['label']).float()
        return {'fnc': fnc, 'sbm': sbm, 'label': label}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    trans = transforms.Compose([transforms.ToTensor()])
    base_path = '..'
    mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')
    dataset = TReNDS_dataset(mat_folder, fnc_path, sbm_path, ICN_num_path, train_scores_path, mask_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_sample in dataloader:
        print(batch_sample)
        break

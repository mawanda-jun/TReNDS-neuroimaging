import os
from h5py import File as h5File
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import torch
import zlib


class TReNDS_dataset(Dataset):
    def __init__(self, mat_folder, transform=None):
        super(Dataset, self).__init__()
        print('Loading dataset...')
        # Load data
        # Store the paths to the .mat file as a dictionary {patientID: complete_path_to_file}
        self.mat_paths = {int(filename.split('.')[0]): os.path.join(mat_folder, filename) for filename in os.listdir(mat_folder)}

        # Prepare num_to_id in order to address the indexes required from torch API
        self.__num_to_id = {i: k for i, k in enumerate(self.mat_paths.keys())}
        # Create reverse order to have control over dataset patients IDs and indexes
        self.id_to_num = {k: i for i, k in self.__num_to_id.items()}

        print('Dataset loaded!')

        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.mat_paths.keys())

    def __getitem__(self, item):
        # Get the ID corresponding to the item (an index) that torch is looking for.
        ID = self.__num_to_id[item]
        # Retrieve all information from the Dataset initialization
        # Keep brain commented until not working on 3D images
        brain = np.array(h5File(self.mat_paths[ID], 'r', rdcc_nbytes=30*1024**2)['SM_feature'])
        # Create sample
        sample = {
            'ID': ID,
            'brain': brain
        }

        # Transform sample (if defined)
        return self.transform(sample) if self.transform else sample


# Custom transform example
class ToTensor:
    def __call__(self, sample):
        # Define use of brain images - which are not necessary for shallow networks
        # Sum all 53 spatial maps into one to make the training lighter
        # If coming from torch tensors, they have already been summed and normalized
        brain = torch.tensor(sample['brain'])

        return {**sample, 'brain': brain}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # trans = transforms.Compose([transforms.ToTensor()])
    base_path = '..'
    train_mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train')
    train_pth_folder = os.path.join(base_path, 'dataset/fMRI_train_torch')
    test_mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_test')
    test_pth_folder = os.path.join(base_path, 'dataset/fMRI_test_torch')
    train_compressed_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train_compressed')
    test_compressed_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_test_compressed')
    # fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    # sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    # train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    dataset = TReNDS_dataset(train_mat_folder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=6)

    mean = torch.zeros(53, 52, 63, 53, device='cuda:0').double()
    variance = torch.zeros_like(mean, device='cuda:0').double()
    i = torch.zeros(1, device='cuda:0').double()
    old_mean = torch.zeros_like(mean, device='cuda:0').double()
    # Save brains into torch format for training set
    for batch in tqdm(dataloader, desc='Converting train brains...'):
        for ID, brain in zip(batch['ID'], batch['brain']):
            brain = brain.to('cuda:0')
            old_mean = mean
            mean += (brain - mean) / i
            variance += (brain - mean) * (brain - old_mean)
            i += 1
            # path = os.path.join(train_compressed_folder, str(ID.item()) + '.gz')
            # brain = zlib.compress(brain.numpy().tobytes(), level=1)
            # open(path, 'wb').write(brain)

    dataset = TReNDS_dataset(test_mat_folder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=6)

    # Save brains into torch format for test set
    for batch in tqdm(dataloader, desc='Converting test brains...'):
        for ID, brain in zip(batch['ID'], batch['brain']):
            brain = brain.to('cuda:0')
            old_mean = mean
            mean += (brain - mean) / i
            variance += (brain - mean) * (brain - old_mean)
            i += 1
            # path = os.path.join(test_compressed_folder, str(ID.item()) + '.gz')
            # brain = zlib.compress(brain.numpy().tobytes(), level=1)
            # open(path, 'wb').write(brain)
    variance = variance / (i - 1)
    torch.save(mean, 'mean.pt')
    torch.save(variance, 'variance.pt')
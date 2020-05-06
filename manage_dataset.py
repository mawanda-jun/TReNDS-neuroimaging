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
        # brain = np.array(h5File(self.mat_paths[ID], 'r', rdcc_nbytes=30*1024**2)['SM_feature'])
        # brain = np.frombuffer(zlib.decompress(open(self.mat_paths[ID], 'rb').read()), dtype='float64').reshape(53, 52, 63, 53)
        brain: np.ndarray = np.array(h5File(self.mat_paths[ID], 'r')['SM_feature'], dtype='float32')
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
        # brain = torch.tensor(sample['brain'], dtype=torch.float32)
        brain = torch.tensor(sample['brain'])

        return {**sample, 'brain': brain}


class Normalize:
    """
    Normalize brain images with mean/std calculated with Welford's algorithm
    """
    def __init__(self, mean_path, std_path):
        self.mean = torch.load(mean_path)
        self.std = torch.load(std_path)
        self.std[self.std == 0] = 100

    def __call__(self, sample, *args, **kwargs):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        brain = sample['brain'].cuda()
        brain = (brain - self.mean) / self.std
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return {**sample, 'brain': brain.cpu().float()}


def welford(dataloader, mean, variance, i):
    # Save brains into torch format for training set
    for batch in tqdm(dataloader, desc='Converting train brains...'):
        for ID, brain in zip(batch['ID'], batch['brain']):
            brain = brain.to('cuda:0')
            old_mean = mean.clone()
            mean += (brain - mean) / i
            variance += (brain - mean) * (brain - old_mean)
            i += 1
    return mean, variance, i


def save(dataloader, dest_path):
    for batch in tqdm(dataloader, desc='Converting train brains...'):
        for ID, brain in zip(batch['ID'], batch['brain']):
            path = os.path.join(dest_path, str(ID.item()) + '.pt')
            torch.save(brain.clone(), path)
            # brain = zlib.compress(brain.numpy().tobytes(), level=1)
            # open(path, 'wb').write(brain)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms


    base_path = '..'
    train_mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train')
    test_mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_test')
    train_torch_folder = os.path.join(base_path, 'dataset/fMRI_train_torch')
    test_torch_folder = os.path.join(base_path, 'dataset/fMRI_test_torch')

    mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    train_trans = transforms.Compose([ToTensor(), Normalize(mean_path, variance_path)])
    test_trans = transforms.Compose([ToTensor()])
    # fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    # sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    # train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    # train_set = TReNDS_dataset(train_mat_folder, transform=train_trans)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=6)
    test_set = TReNDS_dataset(test_mat_folder, transform=test_trans)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=6)

    # mean = torch.zeros(53, 52, 63, 53, device='cuda:0').double()
    # variance = torch.zeros_like(mean, device='cuda:0').double()
    # i = 1

    # mean, variance, i = welford(train_loader, mean, variance, i)
    # mean, variance, i = welford(test_loader, mean, variance, i)

    # variance = variance / (i - 1)
    # torch.save(mean, 'mean.pt')
    # torch.save(variance, 'variance.pt')


    # Save brains into torch format for training set
    # save(train_loader, train_torch_folder)
    save(test_loader, test_torch_folder)
    # Save brains into torch format for test set
    # for batch in tqdm(dataloader, desc='Converting test brains...'):
    #     for ID, brain in zip(batch['ID'], batch['brain']):
            # path = os.path.join(test_compressed_folder, str(ID.item()) + '.mat')
            # brain = zlib.compress(brain.numpy().tobytes(), level=1)
            # open(path, 'wb').write(brain)

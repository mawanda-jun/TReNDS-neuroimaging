import os
from h5py import File as h5File
import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TReNDS_dataset(Dataset):
    def __init__(self, mat_folder, fnc_path, sbm_path, ICN_numbers_path, mask_path, transform=None):
        super(Dataset, self).__init__()
        # Load data
        # Store the paths to the .mat file as a dictionary {ID: complete_path_to_file}
        self.mat_paths = {int(filename.split('.')[0]): os.path.join(mat_folder, filename) for filename in os.listdir(mat_folder)}

        self.fnc = {Id: np.array(pd.read_csv(fnc_path).loc[Id]) for Id in self.mat_paths.keys()}

        self.sbm = {Id: np.array(pd.read_csv(sbm_path).loc[Id]) for Id in self.mat_paths.keys()}

        self.ICN_num = np.array(pd.read_csv(ICN_numbers_path)['ICN_number'])

        self.mask = np.array(nib.load(mask_path))

        self.__num_to_id = {i: k for i, k in enumerate(self.mat_paths.keys())}
        # Prepare num_to_id in order to address the indexes required from torch API

        self.transform = transform

    def __len__(self):
        return len(self.mat_paths.keys())

    def __getitem__(self, idx):
        id = self.__num_to_id[idx]
        brain = np.array(h5File(self.mat_paths[id], 'r')['SM_feature'])
        sample = {
            'filename': self.__data[idx][0],
            'video_embedding': self.__data[idx][1],
            'audio_embedding': self.__data[idx][2],
            'label': self.__data[idx][3]
        }

        # Transform (if defined)
        return self.transform(sample) if self.transform else sample

#
# class RandomCrop:
#     def __init__(self, crop_len):
#         self.crop_len = crop_len
#
#     def __call__(self, sample):
#         video_embedding = sample['video_embedding']
#         # Randomly choose an index
#         tot_frames = video_embedding.shape[0]
#
#         if self.crop_len > tot_frames:
#             cropped_embedding = np.zeros((self.crop_len, video_embedding.shape[1]), dtype='float32')
#             cropped_embedding[0:tot_frames, ...] = video_embedding
#         else:
#             start_idx = int(np.random.random() * (tot_frames - self.crop_len))
#             end_idx = start_idx + self.crop_len
#             cropped_embedding = video_embedding[start_idx:end_idx, ...]
#
#         return {**sample, 'video_embedding': cropped_embedding}
#
#
# class ToTensor:
#     def __call__(self, sample):
#         video_embedding = torch.tensor(sample['video_embedding']).float()
#         audio_embedding = torch.tensor(sample['audio_embedding']).float()
#         # audio_embedding = torch.tensor(np.zeros_like(sample['audio_embedding'])).float()
#         label = torch.tensor(sample['label']).float()
#         return {**sample, 'video_embedding': video_embedding, 'audio_embedding': audio_embedding, 'label': label}
#
#
# class LabelOneHot:
#     def __init__(self):
#         self.l = {
#             0: np.array([0, 1], dtype='uint8'),
#             1: np.array([1, 0], dtype='uint8')
#         }
#
#     def __call__(self, sample):
#         return {**sample, 'label': self.l[sample['label']]}


if __name__ == '__main__':

    # trans = transforms.Compose([RandomCrop(crop_len),
    #                             ToTensor()
    #                             ])
    base_path = '..'
    mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')
    dataset = TReNDS_dataset(mat_folder, fnc_path, sbm_path, ICN_num_path, mask_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_sample in dataloader:
        batch = batch_sample['video_embedding']
        print(batch.shape)
        break

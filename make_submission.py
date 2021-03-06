"""
FILE IS NOT UPDATED WITH THE LAST NETWORK TYPES
"""

from dataset import TReNDS_dataset, ToTensor, Normalize
from vae_classifier import Model

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

if __name__ == '__main__':
    # Define paths
    base_path = '..'
    train_pth_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train_torch')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    run_path = 'experiments/CustomResNet18Siamese_numInitFeatures.16_lr.0.002_lr_decay.1.0_drop.0.4_batchsize.11_loss.metric_optimizer.adamw_patience.10_other_net.new_transforms'
    checkpoint_path = os.path.join(run_path, 'ep_35_checkpoint_0.16796829.pt')
    # Define transformations for files
    # TODO: REMEMBER TO NORMALIZE INPUT
    mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    variance_path = os.path.join(base_path, 'dataset', 'variance.pt')
    trans = transforms.Compose([ToTensor(use_fnc=True, train=False), Normalize(mean_path, variance_path)])

    # Make model starting from folder submission

    model = Model('CustomResNet18Siamese', {'dropout_prob': 0.5, 'num_init_features': 16}, 'adamw', 'metric', lr=1e-4)
    model.net.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.net.to(DEVICE)
    # Create submission
    pth_folder = os.path.join(base_path, 'dataset/fMRI_test_torch')
    test_set = TReNDS_dataset(pth_folder, sbm_path, fnc_path=fnc_path, transform=trans)
    model.submit(DataLoader(test_set, batch_size=64, shuffle=False, pin_memory=True, num_workers=8), run_path)
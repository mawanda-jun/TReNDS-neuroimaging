from dataset import TReNDS_dataset, ToTensor, Normalize
from vae_classifier import Model

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Define paths
    base_path = '..'
    train_pth_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train_torch')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    run_path = 'experiments/PlainResNet3D10_numInitFeatures.128_lr.0.004_lr_decay.0.98_drop.0.4_batchsize.32_loss.metric_optimizer.adamw_patience.10_other_net.normalized'
    checkpoint_path = os.path.join(run_path, 'checkpoint_0.17056873440742493_ep13.pt')
    # Define transformations for files
    # TODO: REMEMBER TO NORMALIZE INPUT
    mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    variance_path = os.path.join(base_path, 'dataset', 'variance.pt')
    trans = transforms.Compose([ToTensor(use_fnc=False, train=False), Normalize(mean_path, variance_path)])

    # Make model starting from folder submission

    model = Model('PlainResNet3D10', {'dropout_prob': 0.5, 'num_init_features': 128}, 'adamw', 'metric', lr=1e-4)
    model.net.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.net.to(DEVICE)
    # Create submission
    pth_folder = os.path.join(base_path, 'dataset/fMRI_test_torch')
    test_set = TReNDS_dataset(pth_folder, sbm_path, transform=trans)
    model.submit(DataLoader(test_set, batch_size=16, shuffle=False, pin_memory=True, num_workers=8), run_path)
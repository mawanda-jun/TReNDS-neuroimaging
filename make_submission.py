from dataset import TReNDS_dataset, ToTensor
from model import Model

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

    run_path = 'experiments/SmartDense3D_loss.metric_batchsize.32_optimizer.adam_lr.0.0001_drop.0.3_patience.5_other_net.1'
    checkpoint_path = os.path.join(run_path, 'checkpoint_0.16984178125858307_ep20.pt')
    # Define transformations for files
    trans = transforms.Compose([ToTensor()])

    # Make model starting from folder submission

    model = Model('SmartDense3D', {'dropout_prob': 0.3}, 'adam', 'metric', lr=1e-4)
    model.net.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.net.to(DEVICE)
    # Create submission
    pth_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_test_torch')
    test_set = TReNDS_dataset(pth_folder, fnc_path, sbm_path, ICN_num_path, None, mask_path, trans)
    model.submit(DataLoader(test_set, batch_size=4, shuffle=False, pin_memory=True, num_workers=8), run_path)
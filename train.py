from model import Model
from dataset import TReNDS_dataset, ToTensor

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

if __name__ == '__main__':
    # Define paths
    base_path = '..'
    mat_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    # Define transformations for files
    trans = transforms.Compose([ToTensor()])

    # Create dataset
    dataset = TReNDS_dataset(mat_folder, fnc_path, sbm_path, ICN_num_path, train_scores_path, mask_path, trans)

    # Split dataset in train/val
    val_dim = 0.3
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)
    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_len, shuffle=False, pin_memory=True)

    # Define net params
    net_params = {
        'fnc_dim': 1378,
        'sbm_dim': 26,
        'dropout_prob': 0.3
    }

    # Define model
    model = Model('ShallowNet', net_params, 'adam', 'metric', lr=1e-4)

    # Train model
    model.fit(5000, train_loader, val_loader, 10)


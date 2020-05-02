from model import Model
from dataset import TReNDS_dataset, ToTensor
import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os


def clean_folder(folder, metric, delta=0.02):
    """
    Cleans all checkpoints that are distant > delta from average metric
    :param folder: folder path
    :return: None
    """
    for file in os.listdir(folder):
        if 'checkpoint' in file:
            filename = file.split('.')[0] + '.' + file.split('.')[1]
            metric_value = filename.split('_')[1]
            metric_value = float(metric_value)
            if not metric - delta < metric_value < metric + delta:
                os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    # Define paths
    base_path = '..'
    train_pth_folder = os.path.join(base_path, 'dataset/Kaggle/fMRI_train_torch')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    # Define transformations for files
    trans = transforms.Compose([ToTensor()])

    # Create dataset
    dataset = TReNDS_dataset(train_pth_folder, fnc_path, sbm_path, ICN_num_path, train_scores_path, mask_path, trans)

    # Split dataset in train/val
    val_dim = 0.3
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define training hyper parameters
    network_type = 'SmartDense3D'
    optimizer = 'adam'
    loss = 'metric'
    learning_rate = 5e-5
    batch_size = 32
    dropout_prob = 0.3
    patience = 5

    # Define network hyper params
    net_hyperparams = {
        'dropout_prob': dropout_prob
    }

    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Define model
    model = Model(network_type, net_hyperparams, optimizer, loss, lr=learning_rate)

    run_path = os.path.join('experiments',
                            network_type +
                            '_loss.' + loss +
                            '_batchsize.' + str(batch_size) +
                            '_optimizer.' + optimizer +
                            '_lr.' + str(learning_rate) +
                            '_drop.' + str(dropout_prob) +
                            '_patience.' + str(patience) +
                            '_other_net.' + str(0))
    os.makedirs(run_path, exist_ok=False)

    # Train model
    val_metric = model.fit(15000, train_loader, val_loader, patience, run_path)
    # Clean checkpoint folder from all the checkpoints that are useless
    clean_folder(run_path, val_metric, delta=0.002)

    # Make backup of network and model files into run folder
    shutil.copy('network.py', run_path)
    shutil.copy('model.py', run_path)
    shutil.copy('train.py', run_path)
    shutil.copy('dataset.py', run_path)
    shutil.copy('pytorchtools.py', run_path)



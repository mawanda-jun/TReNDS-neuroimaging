from vae_classifier import Model
from dataset import TReNDS_dataset, ToTensor, AugmentDataset, fMRI_Aumentation
import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from torch_lr_finder import LRFinder
import json


def clean_folder(folder, metric, delta=0.02):
    """
    Cleans all checkpoints that are distant > delta from average metric
    :param folder: folder path
    :return: None
    """
    for file in os.listdir(folder):
        if 'checkpoint' in file:
            filename = file.split('.')[0] + '.' + file.split('.')[1]  # Keep name and floating value
            metric_value = filename.split('_')[3]  # Select float value
            metric_value = float(metric_value)  # Cast string to float
            if not metric - delta < metric_value < metric + delta:
                os.remove(os.path.join(folder, file))


if __name__ == '__main__':
    # Define paths
    base_path = '..'
    train_torch_folder = os.path.join(base_path, 'dataset/fMRI_train_torch')
    fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    # No need to normalize train set since it has already been normalized while transformed
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define training hyper parameters
    network_type = 'CustomResNet18SiameseMashup'
    optimizer = 'adamw'
    loss = 'metric'
    learning_rate = 3e-3
    learning_rate_decay = .995
    batch_size = 11
    dropout_prob = 0.4
    patience = 10
    num_init_features = 16

    # Define training settings
    lr_range_test = False
    train_workers = 12
    val_workers = 4

    # Define network hyper params
    net_hyperparams = {
        'dropout_prob': dropout_prob,
        'num_init_features': num_init_features  # 64
    }

    # Create dataset
    dataset = TReNDS_dataset(train_torch_folder, sbm_path, fnc_path=fnc_path, train_scores_path=train_scores_path)

    # Split dataset in train/val
    val_dim = 0.3
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define transformations
    train_trans = transforms.Compose([fMRI_Aumentation(), ToTensor(use_fnc=True, train=True, lr_range=lr_range_test)])
    # train_trans = transforms.Compose([ToTensor(use_fnc=True, train=True)])
    val_trans = ToTensor(use_fnc=True, train=True, lr_range=lr_range_test)

    train_set = AugmentDataset(train_set, train_trans)
    val_set = AugmentDataset(val_set, val_trans)

    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=train_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=val_workers)

    # Define model
    model = Model(network_type, net_hyperparams, optimizer, loss, lr=learning_rate, lr_decay=learning_rate_decay)

    if lr_range_test:
        network = model.net
        criterion = model.loss
        optimizer = model.optimizer
        lr_finder = LRFinder(network, optimizer, criterion, device='cuda:0')
        lr_finder.range_test(train_loader, val_loader, end_lr=1., num_iter=100)
        json.dump(lr_finder.history, open('lr_finder.json', 'w'))
        lr_finder.plot()
        lr_finder.reset()

    else:
        run_path = os.path.join('experiments',
                                network_type +
                                '_numInitFeatures.' + str(num_init_features) +
                                '_lr.' + str(learning_rate) +
                                '_lr_decay.' + str(learning_rate_decay) +
                                '_drop.' + str(dropout_prob) +
                                '_batchsize.' + str(batch_size) +
                                '_loss.' + loss +
                                '_optimizer.' + optimizer +
                                '_patience.' + str(patience) +
                                '_other_net.' + 'new_transforms')

        os.makedirs(run_path, exist_ok=False)

        # Make backup of network and model files into run folder
        shutil.copy('network.py', run_path)
        shutil.copy('base_networks.py', run_path)
        shutil.copy('model.py', run_path)
        shutil.copy('train.py', run_path)
        shutil.copy('dataset.py', run_path)
        shutil.copy('pytorchtools.py', run_path)
        shutil.copy('DenseNet3D.py', run_path)
        shutil.copy('ResNet.py', run_path)

        # Train model
        val_metric = model.fit(15000, train_loader, val_loader, patience, run_path)
        # Clean checkpoint folder from all the checkpoints that are useless
        clean_folder(run_path, val_metric, delta=0.002)




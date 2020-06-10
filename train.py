from model import Model
from dataset import \
    TReNDS_dataset, \
    ToTensor, \
    AugmentDataset, \
    fMRI_Aumentation, \
    Normalize, \
    RandomCropToDim, \
    ResizeToDim, \
    ZeroThreshold

from SparseResNet import custom_collate

import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

os.setgid(1000), os.setuid(1000)


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
    base_path = '/opt/dataset'
    train_torch_folder = os.path.join(base_path, 'fMRI_train_norm')
    fnc_path = os.path.join(base_path, 'Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'Kaggle/fMRI_mask.nii')

    # No need to normalize train set since it has already been normalized while transformed
    mean_path = os.path.join(base_path, 'mean.pt')
    variance_path = os.path.join(base_path, 'variance.pt')

    # Define training hyper parameters
    batch_size = 25
    patience = 10

    net_hyperparams = {
        'dropout_prob': 0.3,
        'num_init_features': 8  # 64
    }
    train_params = {
        'base_lr': 1e-5,
        'max_lr': 1e-4,
        'lr': 1e-5,
        'lr_decay': 0.93,
        'use_apex': True,
        'weight_decay': 1e-5,
        'optimizer_type': 'adamw',
        'network_type': 'PlainResNet10BottleneckSiamese',
        'loss_type': 'metric',
    }

    # Define training settings
    train_workers = 6
    val_workers = 6
    val_dim = 0.3
    lr_range_test = True
    if lr_range_test:
        train_workers = 0
    use_fnc = False
    use_sbm = False
    siamese_sparse = False  # HERE
    threshold = 3
    roi_size = (49, 49, 49)

    # Create dataset
    if use_fnc and use_sbm:
        dataset = TReNDS_dataset(train_torch_folder, sbm_path=sbm_path, fnc_path=fnc_path, train_scores_path=train_scores_path)
    elif use_fnc:
        dataset = TReNDS_dataset(train_torch_folder, fnc_path=fnc_path, train_scores_path=train_scores_path)
    elif use_sbm:
        dataset = TReNDS_dataset(train_torch_folder, sbm_path=sbm_path, train_scores_path=train_scores_path)
    else:
        dataset = TReNDS_dataset(train_torch_folder, train_scores_path=train_scores_path)

    # Split dataset in train/val
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define transformations
    common_trans = transforms.Compose([
        ToTensor(use_sbm=use_sbm, use_fnc=use_fnc, train=True, siamese_sparse=False)
    ])
    if siamese_sparse:
        common_trans = transforms.Compose([
            RandomCropToDim(roi_size),
            ZeroThreshold(threshold),
            common_trans
        ])
        collate_fn = custom_collate
    else:
        collate_fn = None
    train_trans = transforms.Compose([fMRI_Aumentation(), common_trans])
    # train_trans = transforms.Compose([ToTensor(use_fnc=True, train=True)])
    val_trans = common_trans

    train_set = AugmentDataset(train_set, train_trans)
    val_set = AugmentDataset(val_set, val_trans)

    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=train_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=val_workers, collate_fn=collate_fn)

    # Define model
    model = Model(net_hyperparams=net_hyperparams, train_params=train_params)

    # Use in case of start training from saved checkpoint
    # import torch
    # last_epoch = 5
    # checkpoint = torch.load('experiments/CustomResNet18Siamese_numInitFeatures.32_lr.0.004_lr_decay.1.0_drop.0.4_batchsize.11_loss.metric_optimizer.adamw_patience.10_other_net.32outputfeatures/ep_5_checkpoint_0.18078171.pt')
    # model.net.load_state_dict(checkpoint['state_dict'])
    # model.optimizer.load_state_dict(checkpoint['optim_state'])
    # if use_apex:
    #     from apex import amp
    #     amp.load_state_dict(checkpoint['apex_state'])

    if lr_range_test:
        from torch_lr_finder import LRFinder
        import json
        network = model.net
        criterion = model.loss
        optimizer = model.optimizer
        lr_finder = LRFinder(network, optimizer, criterion, device='cuda:0')
        lr_finder.range_test(train_loader, val_loader, end_lr=1e-1, num_iter=100)
        json.dump(lr_finder.history, open('lr_finder.json', 'w'))
        lr_finder.plot()
        lr_finder.reset()

    else:
        run_path = os.path.join('experiments',
                                train_params['network_type'] +
                                '_numInitFeatures.' + str(net_hyperparams['num_init_features']) +
                                '_lr.' + str(train_params['lr']) +
                                '_drop.' + str(net_hyperparams['dropout_prob']) +
                                '_batchsize.' + str(batch_size) +
                                '_loss.' + train_params['loss_type'] +
                                '_optimizer.' + train_params['optimizer_type'] +
                                '_patience.' + str(patience) +
                                '_other_net.' + 'simpler_network')

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
        val_metric = model.fit(15000, train_loader, val_loader, patience, run_path, last_epoch=-1)
        # Clean checkpoint folder from all the checkpoints that are useless
        clean_folder(run_path, val_metric, delta=0.002)




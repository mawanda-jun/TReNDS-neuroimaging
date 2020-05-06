from model import Model
from dataset import TReNDS_dataset, ToTensor, Normalize, AugmentDataset, fMRI_Aumentation
import shutil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from torchsummary import summary


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
    train_torch_folder = os.path.join(base_path, 'dataset/fMRI_train_torch')
    # fnc_path = os.path.join(base_path, 'dataset/Kaggle/fnc.csv')
    sbm_path = os.path.join(base_path, 'dataset/Kaggle/loading.csv')
    # ICN_num_path = os.path.join(base_path, 'dataset/Kaggle/ICN_numbers.csv')
    train_scores_path = os.path.join(base_path, 'dataset/Kaggle/train_scores.csv')
    # mask_path = os.path.join(base_path, 'dataset/Kaggle/fMRI_mask.nii')

    # No need to normalize train set since it has already been normalized while transformed
    # mean_path = os.path.join(base_path, 'dataset', 'mean.pt')
    # variance_path = os.path.join(base_path, 'dataset', 'variance.pt')

    # Define transformations
    common_trans = transforms.Compose([ToTensor()])

    # Create dataset
    dataset = TReNDS_dataset(train_torch_folder, sbm_path, train_scores_path=train_scores_path, transform=common_trans)

    # Split dataset in train/val
    val_dim = 0.3
    dataset_len = len(dataset)
    train_len = round(dataset_len * (1-val_dim))
    val_len = round(dataset_len * val_dim)

    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Define augmentations for training dataset
    # TODO: add augmentation to make the training better. However, the model search will be without because not feasible
    train_set = AugmentDataset(train_set, fMRI_Aumentation())

    # Define training hyper parameters
    network_type = 'CustomResNet3D50'
    optimizer = 'adam'
    loss = 'metric'
    learning_rate = 5e-6
    batch_size = 16
    dropout_prob = 0.5
    patience = 20

    # Define network hyper params
    net_hyperparams = {
        'dropout_prob': dropout_prob
    }

    # Define train and val loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=9)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # Define model
    model = Model(network_type, net_hyperparams, optimizer, loss, lr=learning_rate)

    # Print model network
    # summary(model.net, {'sbm': '10001', 'brain': (53, 52, 63, 53)})

    run_path = os.path.join('experiments',
                            network_type +
                            '_loss.' + loss +
                            '_batchsize.' + str(batch_size) +
                            '_optimizer.' + optimizer +
                            '_lr.' + str(learning_rate) +
                            '_drop.' + str(dropout_prob) +
                            '_patience.' + str(patience) +
                            '_other_net.' + 'resnet50')
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
    shutil.copy('DenseNet3D.py', run_path)
    shutil.copy('ResNet.py', run_path)



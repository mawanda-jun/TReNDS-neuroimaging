from base_networks import BaseNetwork
import torch.nn as nn
import sparseconvnet as scn
import torch
import torch.nn.functional as F
from _collections import OrderedDict


class SparseResNet(BaseNetwork):
    def __init__(self, num_init_features, dropout_prob, use_apex, channels=53):
        BaseNetwork.__init__(self, use_apex=use_apex)
        # Define input/output layers dimension
        # spatial_size = self.sparse_model.input_spatial_size(out_size=1)
        spatial_size = (49, 49, 49)
        self.input_layer = scn.InputLayer(3, spatial_size, mode=0)

        # Define network parameters
        features = [num_init_features * 1, num_init_features * 2, num_init_features * 4, num_init_features * 8]
        reps = [2, 2, 2, 2]

        # Define sparse model
        # self.a = scn.SubmanifoldConvolution(
        #     dimension=3,
        #     nIn=channels,  # n input features
        #     nOut=features[0],  # num init features
        #     filter_size=7,
        #     bias=False
        # )
        # self.b = scn.BatchNormReLU(features[0])
        # self.c = scn.MaxPooling(dimension=3, pool_size=3, pool_stride=1)
        # self.d = scn.SparseResNet(3, features[0], [
        #     ['b', features[0], reps[0], 1],  # type, num_features, repetitions, stride
        #     ['b', features[1], reps[1], 2],
        #     ['b', features[2], reps[2], 2],
        #     ['b', features[3], reps[3], 2],
        # ])
        # self.e = scn.Convolution(3, features[3], features[3], 5, 1, False)
        # self.f = scn.BatchNormReLU(features[3])
        # self.g = scn.SparseToDense(3, features[3])

        self.regressor = nn.Sequential(OrderedDict([
            ('act', nn.ReLU(inplace=True)),
            ('drop', nn.Dropout3d(dropout_prob)),
            ('linear', nn.Linear(features[-1], 5))
        ]))

        # Define sparse model
        self.model = scn.Sequential() \
            .add(scn.SubmanifoldConvolution(
            dimension=3,
            nIn=channels,  # n input features
            nOut=features[0],  # num init features
            filter_size=7,
            bias=False
        )) \
            .add(scn.BatchNormReLU(features[0])) \
            .add(scn.MaxPooling(dimension=3, pool_size=3, pool_stride=1)) \
            .add(scn.SparseResNet(3, features[0], [
            ['b', features[0], reps[0], 1],  # type, num_features, repetitions, stride
            ['b', features[1], reps[1], 2],
            ['b', features[2], reps[2], 2],
            ['b', features[3], reps[3], 2],
        ])) \
            .add(scn.Convolution(3, features[3], features[3], 5, 1, False)) \
            .add(scn.BatchNormReLU(features[3])) \
            .add(scn.SparseToDense(3, features[3]))

        self.regressor = nn.Sequential(OrderedDict([
            ('act', nn.ReLU(inplace=True)),
            ('drop', nn.Dropout3d(dropout_prob)),
            ('linear', nn.Linear(features[-1], 5))
        ]))

    def forward(self, inputs):
        indices, values, batch_size = inputs
        brains = self.input_layer([indices, values, batch_size])
        brains = F.relu(self.model(brains))
        # brains = self.a(brains)
        # brains = self.b(brains)
        # brains = self.c(brains)
        # brains = self.d(brains)
        # brains = self.e(brains)
        # brains = self.f(brains)
        # brains = self.g(brains)
        brains = torch.clamp(self.regressor(brains.view(batch_size, -1)), -0.1, 5)
        return brains.exp()

    @staticmethod
    def get_network_inputs(batch, DEVICE):
        indices, values, batch_size = batch[0]
        return indices.to(DEVICE), values.to(DEVICE), batch_size.to(DEVICE)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that retrieves batches and converts them to sparse representation.
        Every batch contains a list of (brain, label), and brain has dimension (53, <spatial_size>).
        <spatial_size> is (52, 63, 53) if not transformed, although (49, 49, 49).
        The aim is to convert those brains into a representation that is good for a sparse network.

        The sparse representation needed in InputLayer inside the SparseResNet network needs coords and values.
        - coords must be N (nonzero elements) x dimension (if <spatial size> = (49, 49, 49), then dimension is 3)
        or
        - coors must be N (nonzero elements) x dimension + 1 (if batch size != 1, and in the last dimension there are the batch index indices)
        - values must be N (nonzero values) x features_dim (is 53, number of channels)
        Therefore, the transformations follows:
        :param batch:
        :return:
        """
        # Remember batch size for later reference
        batch_size = torch.tensor(len(batch), dtype=torch.int16)
        # Prepare the list of labels
        labels = []
        # Prepare empty arrays for indices and values.
        indices_batch = []
        values_batch = []
        # Iterate over the batch
        for j in range(len(batch)):
            # Retrieve brains volume and single brain
            brain = batch[j][0][0]  # Brain has now dimension (53, 49, 49, 49) since the spatial dimension has been cropped.
            # Generate the values. Since there are 53 features, we need to highlight them in the <values> tensor.
            # Move features channel on last position for simplicity in retrieving the values
            brain_channel_last = brain.transpose(0, -1)
            # Calculate nonzero indices
            nz_cl = torch.nonzero(brain_channel_last, as_tuple=True)
            # Retrieve the values. Now values has dimension (49*3, 53)
            values = brain_channel_last[nz_cl[0:-1]]

            # Since there are 53 features, we need to concatenate the indices in a slightly different way then features.
            # Input layer only accept Nxd+1 tensors, so we need to concatenate the features on the space dimension.
            # Calculate nonzero indices for every image of the feature channel
            nonzero_indices = [list(torch.nonzero(a, as_tuple=True)) for a in brain]
            # Attach the batch size indices at the end of each indices.
            for i, _ in enumerate(nonzero_indices):
                nonzero_indices[i].append(torch.full_like(nonzero_indices[i][0], j))
            # Stack the indices with the batch one
            nonzero_indices = [torch.stack(nz, dim=1) for nz in nonzero_indices]
            # Stack all the indices together so the features are distributed over the spatial dimension.
            indices = torch.cat(nonzero_indices, dim=0)
            indices_batch.append(indices)
            values_batch.append(values)
            # Add label to array
            labels.append(batch[j][1])

        indices_batch = torch.cat(indices_batch, dim=0)
        values_batch = torch.cat(values_batch, dim=0)
        images = (indices_batch, values_batch, batch_size)

        labels = torch.stack(labels, dim=0)
        return images, labels


class BaseSiameseSparseNet(BaseNetwork):
    def __init__(self, dropout_prob=0.1, use_apex=False):
        BaseNetwork.__init__(self, use_apex)
        # Define input/output layers dimension
        # spatial_size = self.sparse_model.input_spatial_size(out_size=1)
        spatial_size = (49, 49, 49)
        self.input_layer = scn.InputLayer(3, spatial_size, mode=0)

        # Define regressor
        self.regressor = nn.Sequential(OrderedDict([
            ('drop_s0', nn.Dropout(dropout_prob)),
            ('linear_s0', nn.Linear(self.regressor_features * 53, 2048)),
            ('relu_s0', nn.ReLU(inplace=True)),
            ('drop_s1', nn.Dropout(dropout_prob)),
            ('linear_s1', nn.Linear(2048, 2048)),
            ('relu_s1', nn.ReLU(inplace=True)),
            ('drop_out', nn.Dropout(dropout_prob)),
            ('regressor', nn.Linear(2048, 5))
        ]))

    def forward(self, inputs):
        # Extract brains
        brains = inputs
        batch_size = brains[0][2]
        # VERY IMPORTANT STEP IN THE COMPREHENSION OF SPARSE TENSORS:
        # nonzeros, add batch index to last dimension
        # Go with siamese network, one for each brain's color channel
        # temp = []
        # for indices, values, _ in brains:
        #     x = self.input_layer([indices, values, batch_size])
        #     x = F.relu(self.sparse_model(x), inplace=True)
        #     temp.append(x)
        # brains = torch.cat(temp, dim=1)
        brains = torch.cat(
            [F.relu(self.sparse_model(
                self.input_layer(
                    [indices, values, batch_size]
                ))) for indices, values, _ in brains], dim=1)
        # ALREADY TRIED THIS APPROACH, TESTED OK (BUT NOT FOR BATCHNORM). NO PERFORMANCE INCREASING FOUND.
        # brains = brains.view(brains.size(0)*brains.size(1), brains.size(2), brains.size(3), brains.size(4)).unsqueeze(1)
        # brains = F.relu(self.net_3d(brains))
        # brains = brains.view(inputs[0].size(0), -1)
        # return self.regressor(brains)
        brains = torch.clamp(self.regressor(brains.view(batch_size, -1)), -0.1, 5)
        return brains.exp()

    @staticmethod
    def get_network_inputs(batch, DEVICE):
        return [(b[0].to(DEVICE), b[1].to(DEVICE), b[2].to(DEVICE)) for b in batch[0]]


class CustomSiameseSparseResNet18(BaseSiameseSparseNet):
    def __init__(self, channels=1, num_init_features=64, dropout_prob=0.1, use_apex=False):
        # Define network parameters
        features = [num_init_features * 1, num_init_features * 2, num_init_features * 3, num_init_features * 4]
        reps = [2, 2, 2, 2]
        # Define regressor features for regressor
        self.regressor_features = features[-1]
        # Initialize base class with regressor_features set
        BaseSiameseSparseNet.__init__(self, dropout_prob, use_apex)

        # Define sparse model
        self.sparse_model = scn.Sequential() \
            .add(scn.SubmanifoldConvolution(
            dimension=3,
            nIn=channels,  # n input features
            nOut=features[0],  # num init features
            filter_size=7,
            bias=False
        )) \
            .add(scn.BatchNormReLU(features[0])) \
            .add(scn.MaxPooling(dimension=3, pool_size=3, pool_stride=1)) \
            .add(scn.SparseResNet(3, features[0], [
            ['b', features[0], reps[0], 1],  # type, num_features, repetitions, stride
            ['b', features[1], reps[1], 2],
            ['b', features[2], reps[2], 2],
            ['b', features[3], reps[3], 2],
        ])) \
            .add(scn.Convolution(3, features[3], features[3], 5, 1, False)) \
            .add(scn.BatchNormReLU(features[3])) \
            .add(scn.SparseToDense(3, features[3]))


def custom_siamese_collate(batch):
    """
    Custom collate function that retrieves batches and converts them to sparse representation.
    Every batch contains a list of (brain, label), and brain has dimension (53, <spatial_size>).
    <spatial_size> is (52, 63, 53) if not transformed, although (49, 49, 49).
    The aim is to convert those brains into a representation that is good for a -siamese- sparse network.

    The sparse representation needed in InputLayer inside the SparseResNet network needs coords and values.
    - coords must be N (nonzero elements) x dimension (if <spatial size> = (49, 49, 49), then dimension is 3)
    or
    - coors must be N (nonzero elements) x dimension + 1 (if batch size != 1, and in the last dimension there are the batch index indices)
    - values must be N (nonzero values) x features_dim (is 1, number of channels in the siamese representation).
    In this case, I will return a list of 53 tuples, and for each one there will be the values of the batch images.
    Therefore, the transformations follows:
    :param batch:
    :return:
    """
    # Remember batch size for later reference
    batch_size = torch.tensor(len(batch), dtype=torch.int16)
    # Prepare the list of brains and labels
    images = []
    labels = []
    # Iterate over the channels dimension
    for i in range(53):
        # Prepare empty arrays for indices and values. Those items will be stored separately for each batch.
        indices_batch = []
        values_batch = []
        # Iterate over the batch
        for j in range(len(batch)):
            # Retrieve brains volume and single brain
            brain = batch[j][0][0][i]
            # Find nonzero indices. <as_tuple=True> is needed for advanced indexing, to retrieve the values of indices
            nonzero_indices = list(torch.nonzero(brain, as_tuple=True))
            # Find nonzero values.
            # Values must have the last dimension of the color channel. In this case is 1.
            values = brain[nonzero_indices].unsqueeze(-1)
            # Add batch index to indices tensor. Now tensor has dimension (N, 4) and the last dimension is filled with the batch index
            # This is needed by the InputLayer library. In the last dimension it needs the batch index:
            # Since every item in batch will be concatenated, it must be able to find the right batch item.
            # Stack indices. It will have the representation of (N, 3), which is the number of nonzero indices and the
            # dimension of the spatial size
            nonzero_indices.append(torch.full_like(nonzero_indices[0], j))
            indices = torch.stack(nonzero_indices, -1)
            indices_batch.append(indices)
            values_batch.append(values)
            if i == 0:
                # Add label to array but only once - so in the first pass of images
                labels.append(batch[j][1])

        indices_batch = torch.cat(indices_batch, dim=0)
        values_batch = torch.cat(values_batch, dim=0)
        images.append((indices_batch, values_batch, batch_size))

    labels = torch.stack(labels, dim=0)
    return images, labels


def custom_collate(batch):
    """
    Custom collate function that retrieves batches and converts them to sparse representation.
    Every batch contains a list of (brain, label), and brain has dimension (53, <spatial_size>).
    <spatial_size> is (52, 63, 53) if not transformed, although (49, 49, 49).
    The aim is to convert those brains into a representation that is good for a siamese network. Therefore,
    every item in brain will be transformed into a dimension of (53, batch_size, <spatial_size>).

    The sparse representation needed in InputLayer inside the SparseResNet network needs coords and values.
    - coords must be N (nonzero elements) x dimension (if <spatial size> = (49, 49, 49), then dimension is 3
    - values must be N (nonzero values) x features_dim (for the siamese network is 1, although is 53, number of channels)
    Therefore, the transformations follows:
    :param batch:
    :return:
    """
    # Remember batch size for later reference
    batch_size = torch.tensor(len(batch), dtype=torch.int16)
    # Prepare the list of brains and labels
    labels = []
    # Prepare empty arrays for indices and values. Those items will be stored separately for each batch.
    indices_batch = []
    values_batch = []
    # Iterate over the batch
    for j in range(len(batch)):
        # Retrieve brains volume and single brain
        brain = batch[j][0][0]
        brain_channel_last = brain.transpose(0, -1)
        # Find nonzero indices. <as_tuple=True> is needed for advanced indexing, to retrieve the values of indices
        nonzero_indices = list(torch.nonzero(brain_channel_last, as_tuple=True))
        # Find nonzero values.
        # Values must have the last dimension of the color channel. In this case is 1.
        # In the case of channels, (so dimension 49, 49, 49, 3) this would have been suitable:
        # values = brain[nonzero_indices[0:-1]]
        values = brain_channel_last[nonzero_indices[0:-1]]
        # Add batch index to indices tensor. Now tensor has dimension (N, 4) and the last dimension is filled with the batch index
        # This is needed by the InputLayer library. In the last dimension it needs the batch index:
        # Since every item in batch will be concatenated, it must be able to find the right batch item.
        # Stack indices. It will have the representation of (N, 3), which is the number of nonzero indices and the
        # dimension of the spatial size
        nonzero_indices.append(torch.full_like(nonzero_indices[0], j))
        indices = torch.stack(nonzero_indices, -1)
        indices_batch.append(indices)
        values_batch.append(values)
        # Add label to array but only once - so in the first pass of images
        labels.append(batch[j][1])

    indices_batch = torch.cat(indices_batch, dim=0)
    values_batch = torch.cat(values_batch, dim=0)
    images = (indices_batch, values_batch, batch_size)

    labels = torch.stack(labels, dim=0)
    return images, labels

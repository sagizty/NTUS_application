"""
author Tianyi Zhang

ref:
1. my own repository when learning pytorch
2. arxiv 2006.01561 official code


NOTE
1. the aim of MIL training is to learn how many '7' in the given bag

2. I have already considered the unfull-filled batch as a bag,
by using a zero tensor to fix the size of MLP input

3. distribution pooling mudule from
https://github.com/onermustafaumit/mil_pooling_filters/blob/main/regression/distribution_pooling_filter.py
"""


import os
import math
import gzip
import pickle
import requests
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader


# get MINIST dataset
def make_dataset(PATH, target_nums=(0, 7)):
    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    dataset_train = []
    label_train = []
    for k in range(len(y_train)):
        if y_train[k] in target_nums:
            dataset_train.append(x_train[k])
            label_train.append(y_train[k])
        else:
            pass
    dataset_train = np.array(dataset_train)
    label_train = np.array(label_train)

    dataset_valid = []
    label_valid = []
    for k in range(len(y_valid)):
        if y_valid[k] in target_nums:
            dataset_valid.append(x_valid[k])
            label_valid.append(y_valid[k])
        else:
            pass
    dataset_valid = np.array(dataset_valid)
    label_valid = np.array(label_valid)
    print(dataset_train.shape)
    print(dataset_valid.shape)

    '''
    # check pics
    
    plt.imshow(dataset_train[0].reshape((28, 28)), cmap="gray")
    plt.show()
    plt.close()
    
    plt.imshow(dataset_train[7].reshape((28, 28)), cmap="gray")
    plt.show()
    plt.close()
    '''

    x_train, y_train, x_valid, y_valid = \
        map(torch.tensor, (dataset_train, label_train, dataset_valid, label_valid))

    return x_train, y_train, x_valid, y_valid


def make_dataloader(x_train, y_train, x_valid, y_valid, batch_size=100):
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    return train_dl, valid_dl


def preprocess(x, y, dev):  # Transform MNIST dataset to image
    x_image = torch.cat((x, x, x), 1)
    return x_image.view(-1, 3, 28, 28).to(dev), y.to(dev)


class WrappedDataLoader:  # transform a batch to a bag

    def __init__(self, dl, func, dev, target_nums=(0, 7)):
        self.dl = dl
        self.func = func
        self.dev = dev
        self.target_nums = target_nums

    def __len__(self):
        return len(self.dl)

    def __iter__(self):  # iter
        batches = iter(self.dl)  # get a batch representing a bag

        for x, y in batches:
            # transform and yield a bag,
            # x is a bag of datas
            # y is the number of '7'
            y = torch.sum(y) // self.target_nums[1]
            # FIXME notice last batch is not full, here i altered the MLP projection which makes more sence
            yield self.func(x, y, self.dev)  # fix device


# reprentation loss func
def loss_batch(model, loss_func, xb, yb, opt=None):
    """

    :param model: MIL
    :param loss_func: abasolute loss
    :param xb: a bag (actually a batch) of data
    :param yb: a patch level label but here transforme to a bag level label (by SUM)
    :param opt: optimizer
    :return:
    """
    y = yb
    x = model(xb)
    # print(x.shape)

    # print(x, y)  # check bag level difference

    loss = loss_func(x, y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), 1


def train_one_epoch(epoch, model, loss_func, opt, train_dl, valid_dl, dev=None, writer=None):
    # clean
    running_loss = []
    running_num = []

    # train iteration
    model.train()
    for xb, yb in train_dl:
        xb.to(dev)
        yb.to(dev)
        running_loss_minibatch, running_num_minibatch = loss_batch(model, loss_func, xb, yb, opt)
        running_loss.append(running_loss_minibatch)
        running_num.append(running_num_minibatch)
    train_loss = np.sum(np.multiply(running_loss, running_num)) / np.sum(running_num)

    # val
    model.eval()
    with torch.no_grad():  # 2 ways of coding
        losses, nums = zip(*[loss_batch(model, loss_func, xb.to(dev), yb.to(dev), None) for xb, yb in valid_dl])

    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    if writer is not None:
        # ...log the running loss
        writer.add_scalar('Training loss in each bag',
                          float(train_loss),
                          epoch + 1)
        writer.add_scalar('Validate loss in each bag',
                          float(val_loss),
                          epoch + 1)

    print('\nEpoch:', epoch + 1, '\nTraining loss in each bag:', train_loss, '\nValidate loss in each bag:', val_loss)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev=None, writer=None):
    if dev is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for epoch in range(epochs):
        train_one_epoch(epoch, model, loss_func, opt, train_dl, valid_dl, dev, writer)


# model  CNN backbone
class backbone(nn.Module):  # this is a fake representation of ResNet-18 because the MNIST image is too small

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)

        # a fake residual connection
        self.res_conv = nn.Conv2d(3, 10, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):  # process a bag(as a batch)
        x = x.view(-1, 3, 28, 28)

        x_res = self.res_conv(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x += x_res  # a fake residual connection

        return x


class DistributionPoolingFilter(nn.Module):
    __constants__ = ['num_bins', 'sigma']

    def __init__(self, num_bins=1, sigma=0.1):
        super(DistributionPoolingFilter, self).__init__()

        self.num_bins = num_bins
        self.sigma = sigma
        self.alfa = 1 / math.sqrt(2 * math.pi * (sigma ** 2))
        self.beta = -1 / (2 * (sigma ** 2))

        sample_points = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sample_points', sample_points)

    def extra_repr(self):
        return 'num_bins={}, sigma={}'.format(
            self.num_bins, self.sigma
        )

    def forward(self, data):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


class DistributionWithAttentionPoolingFilter(DistributionPoolingFilter):

    def __init__(self, num_bins=1, sigma=0.1):
        super(DistributionWithAttentionPoolingFilter, self).__init__(num_bins, sigma)

    def forward(self, data, attention_weights):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # data.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        result = self.alfa * torch.exp(self.beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        # attention_weights.size() --> (batch_size,num_instances)
        attention_weights = torch.reshape(attention_weights, (batch_size, num_instances, 1, 1))
        # attention_weights.size() --> (batch_size,num_instances,1,1)
        attention_weights = attention_weights.repeat(1, 1, num_features, self.num_bins)
        # attention_weights.size() --> (batch_size,num_instances,num_features,num_bins)

        result = attention_weights * result
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


# represtation_MLP
class represtation_MLP(nn.Module):
    """
    FFN

    input size of [bag_num,channel,H,W]
    output size of [1,1]

    :param in_features: input neurons
    :param hidden_features: hidden neurons
    :param out_features: output neurons
    :param act_layer: nn.GELU
    :param drop: last drop
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.in_features = in_features

        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.hidden_features)

        self.fc3 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.drop(x)

        return x


class MIL_model(nn.Module):

    def __init__(self, bag_size=100, backbone=backbone(), MIL_pooling=DistributionPoolingFilter()):  #
        super().__init__()
        self.bag_size = bag_size
        self.in_features = 1960  # == channel * H * W
        self.backbone = backbone
        self.MILpool = MIL_pooling
        self.represtation_MLP = represtation_MLP(self.in_features, 100, 1)

    def forward(self, x):
        x = self.backbone(x)
        bag_num, channel, H, W = x.shape
        x = x.flatten(1).unsqueeze(0)  # torch.Size([1, bag_num, self.in_features])

        # appened 0 to fix unfull-bag size problem
        position_size = self.bag_size - bag_num
        x = torch.cat((torch.zeros(1, position_size, self.in_features), x), dim=1)

        x = self.MILpool(x).squeeze(-1)

        x = self.represtation_MLP(x)
        return x


if __name__ == '__main__':

    DATA_PATH = Path("./data")
    PATH = DATA_PATH / "mnist"
    draw_path = os.path.join("./runs")

    if not os.path.exists(draw_path):
        os.makedirs(draw_path)

    writer = SummaryWriter(draw_path)

    # cuda issue
    print('cuda avaliablity:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create the datasets and the dataloaders
    x_train, y_train, x_valid, y_valid = make_dataset(PATH, target_nums=(0, 7))  # use only 0 and 7
    train_dl, valid_dl = make_dataloader(x_train, y_train, x_valid, y_valid, batch_size=100)  # 100 images as a bag

    # DataLoader + mnist transform
    train_dl = WrappedDataLoader(train_dl, preprocess, dev, target_nums=(0, 7))
    valid_dl = WrappedDataLoader(valid_dl, preprocess, dev, target_nums=(0, 7))

    # training
    lr = 0.00001
    epochs = 30
    loss_func = torch.nn.L1Loss(size_average=None, reduce=None)

    model = MIL_model(backbone=backbone(), MIL_pooling=DistributionPoolingFilter()).to(dev)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.005)

    fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev, writer)

    # view tensorboard at --logdir=/Users/zhangtianyi/Study/Torch/MIL/runs --host=0.0.0.0 --port=7777

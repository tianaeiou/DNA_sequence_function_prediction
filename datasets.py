import torch
import h5py
import os
import scipy.io
import numpy as np
import torch.utils.data as Data


def get_train_valid_data(dataset_path):
    np_train_data = h5py.File(os.path.join(dataset_path, "train.mat"))
    np_valid_data = scipy.io.loadmat(os.path.join(dataset_path, "valid.mat"))
    trainX_data = torch.FloatTensor(np.transpose(np.array(np_train_data['trainxdata']), axes=(2, 1, 0)))
    trainY_data = torch.FloatTensor(np.array(np_train_data['traindata']).T)
    validX_data = torch.FloatTensor(np_valid_data['validxdata'])
    validY_data = torch.FloatTensor(np_valid_data['validdata'])
    return trainX_data, trainY_data, validX_data, validY_data


def get_DataLoader(args):
    trainX_data, trainY_data, validX_data, validY_data = get_train_valid_data(args.data_path)
    params = {'batch_size': args.batch_size, 'num_workers': args.num_workers}

    if args.data_type == "mini":
        subset_size = 200
        subset_indices_train = torch.randperm(trainX_data.shape[0])[:subset_size]
        subset_dataset_train = Data.Subset(Data.TensorDataset(trainX_data, trainY_data), subset_indices_train)
        train_loader = Data.DataLoader(subset_dataset_train, shuffle=True, **params)

        subset_indices_valid = torch.randperm(validX_data.shape[0])[:subset_size]
        subset_dataset_valid = Data.Subset(Data.TensorDataset(validX_data, validY_data), subset_indices_valid)
        valid_loader = Data.DataLoader(subset_dataset_valid, shuffle=True, **params)
    else:
        train_loader = Data.DataLoader(dataset=Data.TensorDataset(trainX_data, trainY_data), shuffle=False, **params)
        valid_loader = Data.DataLoader(dataset=Data.TensorDataset(validX_data, validY_data), shuffle=False, **params)
    return train_loader, valid_loader


def get_testDataLoader(args):
    np_test_data = scipy.io.loadmat(os.path.join(args.data_path, "test.mat"))
    testX_data = torch.FloatTensor(np_test_data['testxdata'])
    testY_data = torch.FloatTensor(np_test_data['testdata'])
    params = {'batch_size': args.batch_size, 'num_workers': args.num_workers}

    if args.data_type == "mini":
        subset_size = 200
        subset_indices_test = torch.randperm(testX_data.shape[0])[:subset_size]
        subset_dataset_test = Data.Subset(Data.TensorDataset(testX_data, testY_data), subset_indices_test)
        test_loader = Data.DataLoader(subset_dataset_test, shuffle=True, **params)

    else:
        test_loader = Data.DataLoader(dataset=Data.TensorDataset(testX_data, testY_data), shuffle=False, **params)
    return test_loader

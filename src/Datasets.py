from PIL import Image,ImageChops
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,SubsetRandomSampler
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random


class Cell_Segmentation_Task():

    def __init__(self, root_dir,dataset=None, k_shot=None, transform=None, batch_size=64,online_crop=False):

        self.traindataset = CustomDataset(root_dir, dataset_selection=dataset, k_shot=k_shot,split=True,train_valid='train',
                                          transform=transform)

        self.validdataset = CustomDataset(root_dir, dataset_selection=dataset, k_shot=k_shot, split=True,
                                         train_valid='valid',
                                          transform=transform)
        self.batch_size = batch_size
        self.k_shot = k_shot

        self.images_indices = self.traindataset.selectKshots()

        self.meta_sampler = SubsetRandomSampler(self.images_indices)
        self.train_loader = DataLoader(self.traindataset, batch_size=self.batch_size, sampler=self.meta_sampler)
        self.val_loader = DataLoader(self.validdataset, batch_size=self.batch_size)


class CustomDataset(Dataset):
    def __init__(self, root_dir, dataset_selection, split=False, train_valid='train', target=None,k_shot=None,
                 split_type='train_valid',transform=None):
        self.root_dir = root_dir
        self.selection = dataset_selection
        self.target = target
        self.k_shot = k_shot
        self.transform = transform
        #self.eval = eval
        self.split = split
        self.split_type = split_type
        self.train_valid = train_valid
        self.ground_truth_train = []
        self.ground_truth_valid = []
        self.ground_truth_test = []
        self.images_train = []
        self.images_valid = []
        self.images_test = []

        for set in dataset_selection:
            ground_truth_prefix = set+'/Groundtruth/'
            image_prefix = set+'/Image/'

            if self.split == True:

                if self.split_type == 'train_valid':
                    file_names_g = sorted([self.root_dir + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])
                    file_names_i = sorted(
                        [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                         f[0] != '.'])[:len(file_names_g)]
                    # generate random validtion samples
                    #sample_idxs = random.sample(range(len(file_names_g)), 1000)
                    #f = open('../Datasets/FewShot/Train_Valid_1000/valid_ids_{}.pickle'.format(set), 'wb')
                    #pickle.dump((sample_idxs), f)
                    #f.close()
                    f = open(os.getcwd()+'/Datasets/FewShot/Train_Valid_1000/valid_ids_{}.pickle'.format(set), 'rb')
                    valid_samples = pickle.load(f)
                    f.close()
                    for i in range(len(file_names_g)):
                        if i not in valid_samples:
                            self.ground_truth_train.append(file_names_g[i])
                            self.images_train.append(file_names_i[i])
                        else:
                            self.ground_truth_valid.append(file_names_g[i])
                            self.images_valid.append(file_names_i[i])

                elif self.split_type == 'train_valid_test':
                    self.ground_truth_train =  sorted([self.root_dir + 'Train/' + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + 'Train/' + ground_truth_prefix) if f[0] != '.'])
                    self.images_train = sorted([self.root_dir + 'Train/' +image_prefix + f for f in os.listdir(self.root_dir +'Train/' + image_prefix)
                                                if f[0] != '.'])[:len(self.ground_truth_train)]

                    self.ground_truth_valid = sorted([self.root_dir + 'Valid/' + ground_truth_prefix + f for f in
                                                      os.listdir(self.root_dir + 'Valid/' + ground_truth_prefix) if
                                                      f[0] != '.'])
                    self.images_valid = sorted([self.root_dir + 'Valid/' + image_prefix + f for f in
                                                os.listdir(self.root_dir + 'Valid/' + image_prefix) if
                                                f[0] != '.'])[:len(self.ground_truth_valid)]

                    self.ground_truth_test = sorted([self.root_dir + 'Test/' + ground_truth_prefix + f for f in
                                                      os.listdir(self.root_dir + 'Test/' + ground_truth_prefix) if
                                                      f[0] != '.'])
                    self.images_test = sorted([self.root_dir + 'Test/' + image_prefix + f for f in
                                                os.listdir(self.root_dir + 'Test/' + image_prefix) if
                                                f[0] != '.'])[:len(self.ground_truth_test)]


            else:
                self.ground_truth_train +=  sorted([self.root_dir + ground_truth_prefix + f for f in
                                           os.listdir(self.root_dir + ground_truth_prefix) if f[0] != '.'])
                self.images_train +=  sorted(
                        [self.root_dir + image_prefix + f for f in os.listdir(self.root_dir + image_prefix) if
                         f[0] != '.'])[:len(self.ground_truth_train)]


    def selectKshots(self, splitFactor=1):


        dataset_size = len(self.ground_truth_train)
        indicies = list(range(0, len(self.ground_truth_train)))
        np.random.shuffle(indicies)
        samples = np.random.choice(indicies,self.k_shot if self.k_shot!=None else dataset_size)
        dataSplit = int(np.floor((1-splitFactor) * len(samples)))
        _,trainIdx = samples[:dataSplit], samples[dataSplit:]


        return trainIdx


    def __len__(self):
        if self.split == True:
            if self.train_valid == 'train':
                return(len(self.ground_truth_train))
            elif self.train_valid == 'valid':
                return(len(self.ground_truth_valid))
            elif self.train_valid == 'test':
                return(len(self.ground_truth_test))
        else:
            return(len(self.ground_truth_train))
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.split == True:
            if self.train_valid == 'train':
                image = Image.open(self.images_train[idx])
                ground_truth = Image.open(self.ground_truth_train[idx])
            elif self.train_valid == 'valid':
                image = Image.open(self.images_valid[idx])
                ground_truth = Image.open(self.ground_truth_valid[idx])
            elif self.train_valid == 'test':
                image = Image.open(self.images_test[idx])
                ground_truth = Image.open(self.ground_truth_test[idx])

        else:
            image = Image.open(self.images_train[idx])
            ground_truth = Image.open(self.ground_truth_train[idx])

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5],std=[0.5])])
            ground_truth = transforms.ToTensor()(ground_truth)
            image = transform(image)




        return image,ground_truth

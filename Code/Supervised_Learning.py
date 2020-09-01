import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch,torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from Code import Datasets
from WorkSpace import *
from numpy import Inf
import pickle
import torch.nn as nn
from Code import Models
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import time
class Supervised_Learning():

    def __init__(self,hyperparams=None,targets=None, datasets=None, datasets_path='../Datasets/FewShot/Source/', affine=True,
                 architecture='FCRN',loss='bce',save_dir='/Pre-trained/',experiment_name_postfix=''):

        self.architecture = architecture
        self.hyperparams = hyperparams
        self.datasets = datasets
        self.criterion = loss
        self.affine = affine
        print(loss)
        self.targets=targets
        self.datasets_path = datasets_path
        self.wrkspace = ManageWorkSpace(datasets=datasets)
        self.save_path_prefix = '../models/' + self.wrkspace.map_dict['Supervised_Learning'] + '/'
        self.save_dir = self.save_path_prefix+self.architecture+save_dir
        self.log_path = '../Logging/{}/{}/'.format(self.wrkspace.map_dict['Supervised_Learning'],self.architecture)
        self.experiment_name_postfix = experiment_name_postfix
        self.createDirs()

    def initModel(self):
        return  Models.FCRN(in_channels=1,affine=self.affine,sigmoid=True if self.criterion=='bce' else False) \
            if self.architecture=='FCRN' \
            else Models.UNet(n_class=1,affine=self.affine,sigmoid=True if self.criterion=='bce' else False)

    def createDirs(self):
        k_shot = [1,3,5,7,10]
        self.wrkspace.create_dir([self.save_dir,
                                  self.log_path + 'Pre-trained/Losses/',
                                  self.log_path + 'Trained/Losses/'])
        for shot in k_shot:
            self.wrkspace.create_dir([self.log_path + 'RandomInitFinetune/Losses/'+
                                      str(shot)+'-shot/'])


    def getDataLoader(self,batchsize,dataset_collection,split=True,
                      datasets_path='',split_type='train_valid',shuffle=True,target=None):



        if split==True:
            train_dataset = Datasets.CustomDataset(root_dir=datasets_path, dataset_selection=dataset_collection,
                                                   split=split, split_type=split_type, train_valid='train',
                                                   target=target)
            trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle)

            val_dataset = Datasets.CustomDataset(root_dir=datasets_path, dataset_selection=dataset_collection,
                                                          split=split,split_type=split_type, train_valid='valid',target=target)

            val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=shuffle)

            if split_type=='train_valid_test':
                test_dataset = Datasets.CustomDataset(root_dir=datasets_path, dataset_selection=dataset_collection,
                                                 split=True, split_type=split_type,train_valid='test', target=target)


                test_loader = DataLoader(test_dataset, batch_size=batchsize)
                return trainloader,val_loader,test_loader
            else:
                return trainloader,val_loader
        else:
            train_dataset = Datasets.CustomDataset(root_dir=datasets_path, dataset_selection=dataset_collection)
            val_dataset = Datasets.CustomDataset(root_dir='../Datasets/Supervised_Learning/preprocessed/Valid/',
                                                 dataset_selection=dataset_collection)

            test_dataset = Datasets.CustomDataset(root_dir='../Datasets/Supervised_Learning/preprocessed/Test/',
                                                  dataset_selection=dataset_collection)

            trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle)
            val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=shuffle)

            test_loader = DataLoader(test_dataset, batch_size=batchsize)

            return trainloader,val_loader,test_loader

    def calc_weights(self,labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0,labels.size(0)):
            pos_weight = torch.sum(labels[label_idx]==1)
            neg_weight = torch.sum(labels[label_idx]==0)
            ratio = float(neg_weight.item()/pos_weight.item())
            pos_tensor[label_idx] = ratio*pos_tensor[label_idx]

        return pos_tensor

    def train(self, model,trainloader,val_loader,experiment_name,
              save_every=5,save_dir=None,writer=None):



        optimizer = optim.Adam(params=model.parameters(), lr=self.hyperparams['model_lr'],
                                weight_decay=self.hyperparams['optimizer']['weight_decay'])
        model.cuda()
        train_loss = 0
        train_iou = 0
        train_acc = 0
        val_loss = 0
        val_iou = 0
        val_acc = 0
        total_foreground_train = 0
        total_foreground_val = 0
        val_loss_min = Inf
        train_loss_epoch = []
        val_loss_epoch = []
        epochs = self.hyperparams['epochs']

        start = time.time()
        for e in range(epochs):
            model.train()

            for images, labels in trainloader:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                output,_ = model(images)

                loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output,labels)
                iou_temp,acc_temp = self.intersection_over_union(output,labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                train_iou+=iou_temp
                train_acc+=torch.sum(acc_temp).item()
                total_foreground_train += torch.sum(labels == 1).item()
            print("Time per epoch: ",(time.time()-start)/60)

            if e%20==0:
                output_image_grid = torchvision.utils.make_grid((output.squeeze()>=0.5).type(torch.IntTensor))
                groundtruth_image_grid = torchvision.utils.make_grid(labels.squeeze().type(torch.IntTensor))
                writer.add_images("Prediction Masks "+experiment_name+'_'+str(e+1),(output_image_grid.unsqueeze(dim=1)),e)
                writer.add_images("Groundtruth Masks "+experiment_name+'_'+str(e+1), (groundtruth_image_grid.unsqueeze(dim=1)),e)
            if val_loader != None:
                model.eval()
                for images, labels in val_loader:
                    images, labels = images.cuda(), labels.cuda()
                    with torch.no_grad():
                        output,_ = model(images)
                    iou_temp, acc_temp = self.intersection_over_union(output, labels)
                    if self.criterion=='bce':
                        loss = nn.BCELoss()(output, labels)
                    else:
                        loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output,labels)
                    val_loss += loss.item() * images.size(0)
                    val_iou += iou_temp
                    val_acc += torch.sum(acc_temp).item()
                    total_foreground_val += torch.sum(labels==1).item()

                val_loss = val_loss / len(val_loader.dataset)
                val_iou = val_iou.item() / len(val_loader.dataset)
                val_acc = val_acc / total_foreground_val


                if val_loss <= val_loss_min:
                    print("Saved a better model: {:.6f} ---> {:.6f}".format(val_loss_min, val_loss))
                    torch.save(model.state_dict(),  save_dir+experiment_name+'_state_dict.pt')
                    val_loss_min = val_loss
                val_loss_epoch.append(val_loss)
            else:
                torch.save(model.state_dict(), save_dir + experiment_name + '_state_dict.pt')

            if (e+1) %10 ==0:
                torch.save(model.state_dict(), save_dir + experiment_name +'_'+str(e+1)+ '_state_dict.pt')
            train_loss = train_loss / len(trainloader.dataset)
            train_iou = train_iou.item() / len(trainloader.dataset)
            train_acc = train_acc / total_foreground_train
            print('Epoch:{}//{}\tTrain loss: {:.4f}\tTrain IOU: {:.4f}'
                  '\tTrain Acc: {:.4f}\nVal_loss: {:.4f}\tVal IOU:{:.4f}\tVal Acc:{:.4f}\n'.format(e + 1, epochs, train_loss,train_iou,
                                                                                                 train_acc,val_loss,val_iou,val_acc))


            train_loss_epoch.append(train_loss)

            writer.add_scalars('Train_Val_loss ' + experiment_name,
                               {'train_loss':train_loss,
                                'val_loss':val_loss} ,e)



            val_loss = 0
            val_iou=0
            val_acc=0
            train_loss = 0
            train_acc=0
            train_iou=0
            total_foreground_train=0
            total_foreground_val=0
            if e % save_every == 0 or e + 1 == epochs:
                checkpoint = {'epoch': e,
                              'train_loss': train_loss_epoch,
                              'optimizer': optimizer,
                              'state_dict': model.state_dict()}
                torch.save(checkpoint, save_dir+experiment_name+ '_checkpoint.pt')

        return train_loss_epoch, val_loss_epoch

    def supervised_train(self):
        for target in self.targets:

            datasets = [set for set in self.datasets if set != target]


            print("Source Datasets: {}\tTarget Datasets: {}".format(datasets,target))

            save_dir_dataset = self.save_dir+'Target_'+target+'/'
            self.wrkspace.create_dir([save_dir_dataset])
            experiment_name = 'Supervised_Learning_'+str(self.hyperparams['model_lr'])+'_modellr_'+\
                               str(self.hyperparams['epochs'])+'_epochs_'+target+self.experiment_name_postfix

            model = self.initModel()
            print(model)
            writer = SummaryWriter(log_dir='../Logging/Supervised_Pretraining/'+self.architecture+'/'+experiment_name+'/')
            trainloader,val_loader = self.getDataLoader(batchsize=self.hyperparams['batchsize'],
                                                        dataset_collection=datasets,datasets_path=self.datasets_path,target=target)
            train_loss_epoch, val_loss_epoch = self.train(model=model,trainloader=trainloader,val_loader=val_loader,
                                                          save_dir=save_dir_dataset,experiment_name=experiment_name,writer=writer)
            writer.close()
            result = [train_loss_epoch,val_loss_epoch]
            self.save_results(result,descr='Pre-trained',target=target,
                              filename=experiment_name+'_pretrain_loss_Target_')


    def test(self,model,testloader):
        iou = 0
        foreground_acc = 0
        total_foreground = 0
        test_loss = 0
        model.cuda()
        model.eval()
        for child in model.children():
            if type(child) == nn.Sequential:
                for ii in range(len(child)):
                    if type(child[ii]) == nn.BatchNorm2d:
                        child[ii].track_running_stats = False

        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                output,_ = model(images)

            loss = nn.BCELoss()(output, labels)
            test_loss += loss.item() * images.size(0)
            iou_temp, acc_temp= self.intersection_over_union(output, labels)
            iou+=iou_temp
            foreground_acc+=acc_temp
            total_foreground+=torch.sum(labels==1).item()
        test_loss = test_loss / len(testloader.dataset)
        iou = iou.item() / len(testloader.dataset)
        foreground_acc = torch.sum(foreground_acc).item()/total_foreground
        print('Test Loss: {:.6f} \tTest IOU={:.6f}\tTest FCA={:.6f}\n'.format(test_loss, iou,foreground_acc))
        return test_loss,iou,foreground_acc

    def intersection_over_union(self, tensor, labels, device=torch.device("cuda:0")):
        iou = 0
        foreground_acc = 0
        labels_tens = labels.type(torch.BoolTensor)
        ones_tens = torch.ones_like(tensor, device=device)
        zeros_tens = torch.zeros_like(tensor, device=device)
        if tensor.shape[0] > 1:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum((1, 2))

            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum((1, 2))
            iou += torch.sum((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc+=intersection_tens
        else:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum()
            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum()
            iou += torch.sum((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens

        del temp_tens
        del labels_tens
        del ones_tens
        del zeros_tens
        torch.cuda.empty_cache()
        total_iou = iou
        return total_iou,foreground_acc

    def save_results(self,results,target,descr='',filename=''):
        f_csv = open(self.log_path+descr+'/Losses/'+ filename + target + '.csv', 'w')
        f_pickle = open(self.log_path + descr + '/Losses/' + filename + target + '.pickle', 'wb')
        pickle.dump(results,f_pickle)
        df = pd.DataFrame(results)
        df.to_csv(f_csv,header=False,index=False)
        f_csv.close()
        f_pickle.close()

if __name__ == '__main__':

    datasets = ['B5','B39','EM','ssTEM','TNBC']
    target = ['TNBC']
    hyperparams = {'model_lr':0.01,
                   'epochs':1,
                   'batchsize':64,
                   'optimizer':{'weight_decay':0.0005,
                                'momentum':0.9}
                   }


    supervised_learn = Supervised_Learning(hyperparams=hyperparams, datasets=datasets, save_dir='/Pre-trained/',loss='weightedbce',
                                            datasets_path='../Datasets/FewShot/Source/',targets=target,architecture='UNet',experiment_name_postfix='test')

    supervised_learn.supervised_train()
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from WorkSpace import *
import torch
import torch.optim as optim
from Code import Reptile
from torch.utils.tensorboard import SummaryWriter

class Meta_Learning():

    def __init__(self, hyperparams=None, online_crop=False, datasets_path=os.getcwd()+'/Datasets/FewShot/Source/',
                 datasets=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],target_datasets=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],
                 experiment_name_postfix='',affine=False,
                 methods=None, architecture='FCRN',loss='bce'):

        if datasets is None:
            datasets = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']
        if methods is None:
            methods = ['BCE','BCE_Entropy', 'BCE_Distillation', 'Combined']
        if hyperparams is None:
            hyperparams = {'meta_lr': 0.001,
                   'meta_epochs': 300,
                   'model_lr': 0.01,
                   'inner_epochs': 20,
                   'k-shot': 5,
                   'optimizer': {'weight_decay': 0.0005,
                                 'momentum': 0.9}}
        self.architecture = architecture
        self.affine = affine
        self.criterion = loss
        self.methods = methods
        self.hyperparams = hyperparams
        self.target_datasets = target_datasets
        self.datasets=datasets
        self.datasets_path = datasets_path
        self.online_crop = online_crop
        self.wrkspace = ManageWorkSpace(datasets=datasets)
        self.save_path_prefix = os.getcwd()+'/models/{}/{}/'.format(self.wrkspace.map_dict['Meta_Learning'],self.architecture)
        self.experiment_name_postfix=experiment_name_postfix
        self.createModelsDir()



    def createModelsDir(self):
        for method in self.methods:
            self.wrkspace.create_dir([self.save_path_prefix,
                                      self.save_path_prefix + '/Pre-trained/',
                                      self.save_path_prefix + '/Pre-trained/'+method])

    def createExperimentDirs(self, method, target, experiment_name):
        prefix = self.save_path_prefix + 'Pre-trained/'+method+ '/'
        self.wrkspace.create_dir([prefix + 'Target_' + target + '/',
                                  prefix + 'Target_' + target + '/' + experiment_name,
                                  prefix + 'Target_' + target + '/' + experiment_name + '/State_Dict/',
                                  prefix + 'Target_' + target + '/' + experiment_name + '/Losses_bce/',
                                  prefix + 'Target_' + target + '/' + experiment_name + '/Losses_entropy/',
                                  prefix + 'Target_' + target + '/' + experiment_name + '/Losses_KD/'])


    def getExperimentName(self,method,dataset):
        meta_lr, meta_epochs = self.hyperparams['meta_lr'], self.hyperparams['meta_epochs']
        lr, epochs = self.hyperparams['model_lr'], self.hyperparams['inner_epochs']
        k_shot = self.hyperparams['k-shot']

        experiment_name = 'Meta_Learning_' + method + '_' + str(meta_lr) + \
                          'meta_lr_' + str(lr) + 'modellr_' + str(meta_epochs) + \
                          'meta_epochs_' + str(epochs) + 'inner_epochs_' + str(k_shot) + \
                          'shot_' + dataset+self.experiment_name_postfix

        return experiment_name

    def meta_train(self):
        print(self.hyperparams)
        for method in self.methods:
            methods_map = {'BCE': True, 'BCE_Entropy': False, 'BCE_Distillation': False, 'Combined': False,
                           method: True}
            if method=='Combined':
                methods_map['BCE_Distillation']=True
                methods_map['BCE_Entropy']=True

            print("Training with {} method".format(method))

            for target_dataset in self.target_datasets:
                print("Source Datasets: {}\tTarget Dataset: {}".format([set for set in self.datasets if set != target_dataset],
                                                                    target_dataset))

                experiment_name = self.getExperimentName(method=method,dataset=target_dataset)
                writer = SummaryWriter(log_dir=os.getcwd()+'/Logging/Meta_Training/'+self.architecture+'/'+method+'/'+experiment_name+'/')
                self.createExperimentDirs(method, target_dataset, experiment_name)
                algo = Reptile.Reptile(architecture=self.architecture, hyperparameters=self.hyperparams,datasets=self.datasets,
                                       experiment_name=experiment_name,writer=writer,
                                       datasets_path=self.datasets_path, num_tasks=len(self.datasets),affine=self.affine,
                                       target_dataset=target_dataset,loss=self.criterion, entropy_loss=methods_map['BCE_Entropy'],
                                       dist_loss=methods_map['BCE_Distillation'],
                                       save_folder=self.save_path_prefix + 'Pre-trained/'+method + \
                                                   '/Target_' + target_dataset + '/' + experiment_name)


                algo.outer_segment_loop()
                writer.close()



if __name__ == '__main__':
    datasets = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']
    hyperparams = {'meta_lr': 1.0,
                   'meta_epochs': 700,
                   'model_lr': 0.001,
                   'inner_epochs': 30,
                   'alpha': 0.1,
                   'beta': 1.0,
                   'k-shot': 5,
                   'optimizer': {'weight_decay': 0.0005,
                                 'momentum': 0.9}}

    methods = ['BCE']
    datasets_path = os.getcwd()+'/Datasets/FewShot/Source/'
    meta_learn = Meta_Learning(hyperparams=hyperparams,target_datasets=['EM'],
                               datasets=datasets, methods=methods,datasets_path=datasets_path,loss='weightedbce', affine=False,
                               architecture='FCRN',experiment_name_postfix='test')

    meta_learn.meta_train()

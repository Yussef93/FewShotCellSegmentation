import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from Code import Datasets,Models
import torchvision
import torch.nn.functional as F
import os

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = -1*torch.sigmoid(x) * F.logsigmoid(x)
        b = b.sum()/(b.size(0)*b.size(2)*b.size(3))

        return b
class Reptile():

    def __init__(self, hyperparameters,meta_batch_size=10,num_tasks=None,datasets=None,affine=False,
                 datasets_path=os.getcwd()+'/Dataset/', experiment_name='', save_folder='',
                 architecture='FCRN', target_dataset=None,loss='bce',
                 entropy_loss=False, dist_loss = False,writer=None):

        self.writer = writer
        self.criterion=loss
        self.affine = affine
        print(self.criterion)
        self.entropy_loss = entropy_loss
        self.dist_loss = dist_loss
        self.architecture = architecture
        self.experiment_name = experiment_name
        self.datasets = datasets
        self.target_dataset = target_dataset

        self.model = self.generate_meta_model()

        if torch.cuda.is_available():
            self.model.cuda()

        self.num_inner_epochs = hyperparameters['inner_epochs']
        self.model_lr = hyperparameters['model_lr']
        self.model_weight_decay = hyperparameters['optimizer']['weight_decay']
        self.meta_step_size = hyperparameters['meta_lr']
        self.k_shot = hyperparameters['k-shot']
        self.momentum = hyperparameters['optimizer']['momentum']
        self.alpha = hyperparameters['alpha']
        self.beta = hyperparameters['beta']

        self.num_meta_epochs = hyperparameters['meta_epochs']
        self.meta_batch_size = meta_batch_size

        self.number_tasks = num_tasks

        self.datasets_path = datasets_path
        self.experiment_name = experiment_name
        self.save_folder = save_folder

    def generate_meta_model(self):

        if self.architecture=='FCRN':
            self.model = Models.FCRN(in_channels=1,affine=self.affine,sigmoid=True if self.criterion=='bce' else False)
        elif self.architecture=='UNet':
            self.model = Models.UNet(n_class=1,affine=self.affine,sigmoid=True if self.criterion=='bce' else False)

        print("-----------Meta Model-----------\n", self.model)
        print("--------------------------------\n")
        return self.model


    def gen_segment_datasets(self):
        source_tasks = []
        for set in self.datasets:
            if set != self.target_dataset:
                source_tasks.append(set)
        return source_tasks

    def calc_weights(self,labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0,labels.size(0)):
            pos_weight = torch.sum(labels[label_idx]==1)
            neg_weight = torch.sum(labels[label_idx]==0)
            ratio = float(neg_weight.item()/pos_weight.item())
            pos_tensor[label_idx] = ratio*pos_tensor[label_idx]

        return pos_tensor



    def inner_segment_loop(self, S_tau, meta_epoch,S_p=None,S_n=None):

        tmp_model = Models.FCRN(in_channels=1, affine=self.affine,
                                sigmoid=True if self.criterion == 'bce' else False) \
            if self.architecture == 'FCRN' \
            else Models.UNet(n_class=1,affine=self.affine,
                                sigmoid=True if self.criterion == 'bce' else False)

        for p, q in zip(tmp_model.parameters(), self.model.parameters()):
            p.data = q.clone()


        optimizer = optim.Adam(params=tmp_model.parameters(), lr=self.model_lr,
                               weight_decay=self.model_weight_decay)
        if torch.cuda.is_available():
            tmp_model.cuda()


        Sample_Task1 = Datasets.Cell_Segmentation_Task(self.datasets_path,dataset=[S_tau],k_shot=self.k_shot,transform=None)
        if self.entropy_loss or self.dist_loss:
            Sample_Task2 = Datasets.Cell_Segmentation_Task(self.datasets_path, dataset=[S_p], k_shot=self.k_shot, transform=None)
            if self.entropy_loss and self.dist_loss:
                Sample_Task3 = Datasets.Cell_Segmentation_Task(self.datasets_path, dataset=[S_n], k_shot=self.k_shot,
                                                               transform=None)
            else:
                Sample_Task3 = Sample_Task2

        for epoch in range(0,self.num_inner_epochs):
            tmp_model.train()
            optimizer.zero_grad()
            loss = 0
            bceloss = 0
            entropyloss = 0
            dist_loss = 0
            for image, groundtruth_task1 in Sample_Task1.train_loader:
                if torch.cuda.is_available():
                    image, groundtruth_task1 = image.cuda(),groundtruth_task1.cuda()

                output_task1,temp = tmp_model(image)


                if self.criterion!='bce':
                    bceloss += nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(groundtruth_task1))(output_task1,groundtruth_task1)
                else:
                    bceloss += nn.BCELoss()(output_task1, groundtruth_task1)

            loss += bceloss
            output_dist_task1=temp
            if self.entropy_loss:

                for image,_ in Sample_Task3.train_loader:
                    image = image.cuda()

                    output, _ = tmp_model(image)


                    entropyloss += self.alpha*HLoss()(output)

                loss += entropyloss
            if self.dist_loss:
                for image, _ in Sample_Task2.train_loader:
                    image = image.cuda()

                    _, output_dist_task2 = tmp_model(image)

                    dist_loss += self.beta*nn.MSELoss()(output_dist_task1,
                                              output_dist_task2)
                loss += dist_loss

            loss.backward()
            optimizer.step()
            if epoch == (self.num_inner_epochs - 1):
                # save the final loss after num_inner_epochs
                self.Meta_Losses_bce[meta_epoch] += bceloss.item()
                if self.entropy_loss:
                    self.Meta_Losses_entropy[meta_epoch] += entropyloss.item()
                if self.dist_loss:
                    self.Meta_Losses_KD[meta_epoch] += dist_loss.item()

            if ((meta_epoch+1==50) or ((meta_epoch+1)%100==0)) and (epoch == (self.num_inner_epochs - 1)):
                output_image_grid = torchvision.utils.make_grid((output_task1.squeeze() >= 0.5).type(torch.IntTensor))
                groundtruth_image_grid = torchvision.utils.make_grid(groundtruth_task1.squeeze().type(torch.IntTensor))
                self.writer.add_images("Prediction Masks " + self.experiment_name+'_'+str(meta_epoch+1)+'_'+S_tau, (output_image_grid.unsqueeze(dim=1)), meta_epoch)
                self.writer.add_images("Groundtruth Masks " + self.experiment_name+'_'+str(meta_epoch+1)+'_'+S_tau, (groundtruth_image_grid.unsqueeze(dim=1)),
                                       meta_epoch)


        return list(tmp_model.parameters())

    def outer_segment_loop(self,save_every=1):
        val_loss_epoch = []
        val_loss_min = np.Inf
        val_loss = 0
        self.training_times = torch.zeros(self.num_meta_epochs)
        self.Meta_Losses_bce = torch.zeros(self.num_meta_epochs)
        self.Meta_Losses_entropy = torch.zeros(self.num_meta_epochs)
        self.Meta_Losses_KD = torch.zeros(self.num_meta_epochs)
        self.val_loss = torch.zeros(self.num_meta_epochs)
        for meta_epoch in range(self.num_meta_epochs):
            start = time.time()

            meta_model_params = [- w.clone() for w in self.model.parameters()]
            sum = [- w.clone()*0 for w in self.model.parameters()]

            List_of_Datasets = self.gen_segment_datasets()
            print('----------\nMeta Epoch: ', meta_epoch + 1, '//', self.num_meta_epochs,'----------\n')
            task_num = 0
            for S_m in List_of_Datasets:
                task_num += 1
                data_subset = [dataset for dataset in List_of_Datasets if dataset!=S_m]
                S_p = np.random.choice(data_subset)
                S_n = np.random.choice(data_subset)

                task_weights = self.inner_segment_loop(S_tau=S_m,meta_epoch=meta_epoch,S_p=S_p,S_n=S_n)

                time_start = time.time()

                for k,p, q in zip(sum, meta_model_params, task_weights):
                    k.data += (q.data+p.data)

                self.training_times[meta_epoch] += (time.time() - time_start)

                del (task_weights)
                time_start = time.time()


                self.training_times[meta_epoch] += (time.time() - time_start)


            for weight_old, update in zip(self.model.parameters(), sum):
                weight_old.data += (self.meta_step_size / len(List_of_Datasets)) * update  # meta update



            if (meta_epoch + 1) % save_every == 0:

                torch.save(self.Meta_Losses_bce, self.save_folder + '/Losses_bce/Loss_Step_' + str(meta_epoch + 1))
                ValTasks=None
                if ValTasks!=None:
                    self.model.eval()
                    for images, labels in ValTasks.val_loader:
                        images, labels = images.cuda(), labels.cuda()
                        with torch.no_grad():
                            output, _ = self.model(images)

                        if self.criterion == 'bce':
                            loss = nn.BCELoss()(output, labels)
                        else:
                            loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output, labels)
                        val_loss += loss.item() * images.size(0)


                    val_loss = val_loss / len(ValTasks.val_loader.dataset)
                    self.val_loss[meta_epoch] = val_loss

                    if val_loss <= val_loss_min:
                        print("Saved a better model: {:.6f} ---> {:.6f}".format(val_loss_min, val_loss))
                        torch.save(self.model.state_dict(),
                                   self.save_folder + '/State_Dict/' + self.experiment_name + '_state_dict.pt')
                        val_loss_min = val_loss
                    val_loss_epoch.append(val_loss)
                    val_loss = 0
                    self.model.train()


                if (meta_epoch+1) % 100 ==0:
                    torch.save(self.model.state_dict(),
                               self.save_folder + '/State_Dict/' + self.experiment_name + '_'+str(meta_epoch+1)+'_state_dict.pt')



            self.Meta_Losses_bce[meta_epoch] = self.Meta_Losses_bce[meta_epoch] / len(List_of_Datasets)
            self.Meta_Losses_entropy[meta_epoch] = self.Meta_Losses_entropy[meta_epoch] / len(List_of_Datasets)
            self.Meta_Losses_KD[meta_epoch] = self.Meta_Losses_KD[meta_epoch] / len(List_of_Datasets)


            if self.entropy_loss and self.dist_loss:
                torch.save(self.Meta_Losses_entropy,
                           self.save_folder + '/Losses_entropy/Loss_Step_' + str(meta_epoch + 1))
                torch.save(self.Meta_Losses_KD, self.save_folder + '/Losses_KD/Loss_Step_' + str(meta_epoch + 1))
                self.writer.add_scalars('Meta Losses '+self.experiment_name, {'Loss_entropy':self.Meta_Losses_entropy[meta_epoch],
                                                        'Loss_KD':self.Meta_Losses_KD[meta_epoch],
                                                        'Loss_BCE':self.Meta_Losses_bce[meta_epoch]}, meta_epoch)


            elif self.dist_loss:
                torch.save(self.Meta_Losses_KD, self.save_folder + '/Losses_KD/Loss_Step_' + str(meta_epoch + 1))
                self.writer.add_scalars('Meta Losses ' + self.experiment_name,
                                        {'Loss_KD': self.Meta_Losses_KD[meta_epoch],
                                         'Loss_BCE': self.Meta_Losses_bce[meta_epoch]}, meta_epoch)
            elif self.entropy_loss:
                torch.save(self.Meta_Losses_entropy,
                          self.save_folder + '/Losses_entropy/Loss_Step_' + str(meta_epoch + 1))
                self.writer.add_scalars('Meta Losses ' + self.experiment_name,
                                        {'Loss_entropy': self.Meta_Losses_entropy[meta_epoch],
                                         'Loss_BCE': self.Meta_Losses_bce[meta_epoch]}, meta_epoch)
            else:
                self.writer.add_scalars('Meta_Train_BCE ' + self.experiment_name,
                                   {'train_loss': self.Meta_Losses_bce[meta_epoch]}, meta_epoch)


if __name__ == "__main__":

    hyperparams = {'meta_lr': 1.0,
                   'meta_epochs': 300,
                   'model_lr': 0.001,
                   'inner_epochs': 30,
                   'alpha': 0.01,
                   'beta': 0.01,
                   'k-shot': 5,
                   'optimizer': {'weight_decay': 0.0005,
                                 'momentum': 0.9}}
    datasets_path = '../Datasets/FewShot/Source/'
    datasets = ['B5', 'B39', 'EM', 'ssTEM', 'TNBC']

    algo = Reptile(architecture='FCRN', hyperparameters=hyperparams,
                   experiment_name='Test', num_tasks=len(datasets),
                   target_dataset='TNBC', entropy_loss=False,
                   dist_loss=False,
                   save_folder='../models/Meta-models/BCE/Test')
    algo.outer_segment_loop()

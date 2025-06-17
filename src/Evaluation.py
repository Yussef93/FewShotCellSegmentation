from WorkSpace import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pickle
from Code import Models, Datasets,Results
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import os
class Evaluation():

    def __init__(self, evaluation_config, lr_method=None, switch_affine=False,state_dict_epoch=None,
                 meta_params=None, supervised_params=None,loss_function='bce',affine=False,
                 architecture='FCRN', fig_params=None, pretrained_experiment_name_postfix='',ft_experiment_name_postfix=''):

        self.wrkspace = ManageWorkSpace(datasets=evaluation_config['targets'])
        self.meta_params = meta_params
        self.supervised_params = supervised_params
        self.lr_method = lr_method
        self.state_dict_epoch = state_dict_epoch
        self.evaluation_config = evaluation_config
        self.architecture = architecture
        self.fig_params = fig_params
        self.criterion=loss_function
        self.affine = affine
        self.switch_affine = switch_affine
        self.few_shot_target_dir = os.getcwd()+'/Datasets/FewShot/Target2/'
        self.pretrained_experiment_name_postfix = pretrained_experiment_name_postfix
        self.ft_experiment_name_postfix = ft_experiment_name_postfix
        self.results = Results.Results(lr_method=self.lr_method,meta_methods=self.meta_params['methods'],
                               evaluation_config={'targets':self.evaluation_config['targets'],'selections':self.evaluation_config['selections'],
                                                  'k-shot':self.evaluation_config['k-shot']})
        self.createFineTuneDirs()

    def createFineTuneDirs(self):
        """
        create main dirs for storing fine-tuned models and results

        """
        map_lr_method = self.wrkspace.map_dict[self.lr_method]
        models_save_dir = os.getcwd()+'/models/' + map_lr_method + '/' + self.architecture + '/'
        logging_save_dir = os.getcwd()+'/Logging/' + map_lr_method + '/' + self.architecture + '/'
        self.wrkspace.create_dir([models_save_dir + 'Fine-tuned/',
                                  logging_save_dir + 'Fine-tuned/'])

    def creatEvaluationMetaDirs(self):
        """
          create sub dirs for storing fine-tuned meta-models and results

         """
        model_save_dir = os.getcwd()+'/models/Meta-models/' + self.architecture + '/'
        logging_save_dir = os.getcwd()+'/Logging/Meta-models/' + self.architecture + '/'

        for k_shot in self.evaluation_config['k-shot']:
            for meta_method in self.meta_params['methods']:
                for target in self.evaluation_config['targets']:
                    self.wrkspace.create_dir([model_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/',
                                              logging_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/FT_Loss_IoU/',
                                              logging_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/Test_Loss_IoU/'])
        return model_save_dir, logging_save_dir

    def createEvaluationTransferDirs(self):
        """
        create sub dirs for saving transfer learning models and results

        """
        model_save_dir = os.getcwd()+'/models/Supervised-models/' + self.architecture + '/'
        logging_save_dir = os.getcwd()+'/Logging/Supervised-models/' + self.architecture + '/'

        for k_shot in self.evaluation_config['k-shot']:
            for target in self.evaluation_config['targets']:
                self.wrkspace.create_dir([model_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/Target_' + target + '/',
                                          logging_save_dir + 'Fine-tuned/' + str(
                                              k_shot) + '-shot/Target_' + target + '/FT_Loss_IoU/',
                                          logging_save_dir + 'Fine-tuned/' + str(
                                              k_shot) + '-shot/Target_' + target + '/Test_Loss_IoU/'])
        return model_save_dir, logging_save_dir

    def initModel(self):
        """
        creates an instance of deep neural network acrhitecture (FCRN or UNet)
        :return: randomly initilaized network
        """
        return Models.FCRN(in_channels=1, affine=self.affine, sigmoid=True if self.criterion == 'bce' else False) \
            if self.architecture == 'FCRN' \
            else Models.UNet(n_class=1, affine=self.affine, sigmoid=True if self.criterion == 'bce' else False)

    def load_model_state_dict(self, state_dict_path,epoch=None):
        """
        :param state_dict_path: path to saved pre-trained parameters
        :param epoch: load saved pre-trained parameters from training epoch #
        :return: pre-trained model
        """
        model = self.initModel()
        if epoch==None:
            model.load_state_dict(torch.load(state_dict_path + '_state_dict.pt'))
        else:
            model.load_state_dict(torch.load(state_dict_path +'_' +str(epoch) +'_state_dict.pt'))
        return model

    def getFTandTestLoader(self, selection_ft_path, selection_test_path, batchsize_ftset,
                           batchsize_testset, dataset):

        """

        :param selection_ft_path: path to fine-tune (few-shot) samples from target selection #
        :param selection_test_path: path to test samples from target selection #
        :param batchsize_ftset: batch size of fine-tuning set
        :param batchsize_testset: batch size of test set
        :param dataset: target data set
        :return: finetuneloader, testloader
        """
        finetune_set = Datasets.CustomDataset(root_dir=selection_ft_path, dataset_selection=[dataset])
        testset = Datasets.CustomDataset(root_dir=selection_test_path, dataset_selection=[dataset])
        finetuneloader = DataLoader(finetune_set, batch_size=batchsize_ftset, shuffle=True)
        testloader = DataLoader(testset, batch_size=batchsize_testset[dataset])

        return finetuneloader, testloader

    def getExperimentName(self, k_shot='', target='', descr='finetuned',
                          selection='', lr_method=None):
        prefix = ''
        if lr_method == 'Meta_Learning':
            prefix = lr_method + '_' + descr + '_' + str(self.meta_params['hyperparams']['meta_lr']) + 'meta_lr_' + \
                     str(self.meta_params['hyperparams']['meta_epochs']) + 'meta_epochs_' + \
                     str(self.meta_params['hyperparams']['model_lr']) + 'model_lr_' + \
                     str(self.meta_params['hyperparams']['inner_epochs']) + 'inner_epochs_' + \
                     str(self.meta_params['hyperparams']['k-shot']) + 'shot' + self.ft_experiment_name_postfix


        elif lr_method == 'Transfer_Learning':
            prefix = lr_method + '_' + descr + '_' + self.supervised_params['model_lr'] + 'model_lr_' + \
                     self.supervised_params['epochs'] + '_epochs_' + self.ft_experiment_name_postfix


        if descr == 'finetuned':
            prefix = prefix + '_' + str(self.evaluation_config['ft_epochs']) + '_ft_epochs_' + \
                     str(self.evaluation_config['ft_lr']) + 'ft_lr'



        experiment_name = prefix + str(k_shot) + 'shot_' + target + '_Selection_' + str(selection)

        return experiment_name, prefix

    def getPretrainedMetaModelName(self, meta_method, target):
        """
        Gets name of pre-trained meta-model i.e pretrainedid
        :param meta_method: meta-training method i.e using BCE Loss (BCE) only or BCE+ER (BCE_Entropy),
        BCE+KD (BCE_Distillation), BCE+ER+KD (Combined)
        :param target: name of target data set
        :return: pre--trained model's name
        """
        meta_pre_train_hyperparams = self.meta_params['hyperparams']
        prefix = 'Meta_Learning_' + meta_method + '_' + meta_pre_train_hyperparams['meta_lr'] + \
                 'meta_lr_' + meta_pre_train_hyperparams['model_lr'] + 'modellr_' + meta_pre_train_hyperparams[
                     'meta_epochs'] + \
                 'meta_epochs_' + meta_pre_train_hyperparams['inner_epochs'] + 'inner_epochs_' + \
                 meta_pre_train_hyperparams['k-shot'] + 'shot_'
        model_name = prefix + target + self.pretrained_experiment_name_postfix
        return model_name

    def getTargetandFTDir(self, selection_dir, k_shot):
        target_ft_dir = selection_dir + 'FinetuneSamples/' + str(k_shot) + '-shot/preprocessed/'
        target_test_dir = selection_dir + 'TestSamples/' + str(k_shot) + '-shot/'
        return target_ft_dir, target_test_dir

    def swithBatchNormAffine(self,model):
        for m in model.modules():
            if isinstance(m, nn.Sequential):
                m[1] = nn.BatchNorm2d(m[1].num_features, affine=True)
                if self.architecture == 'UNet':
                    m[4] = nn.BatchNorm2d(m[4].num_features, affine=True)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                Models.init.constant_(m.weight, 0.1)
                Models.init.constant_(m.bias, 0)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert (torch.equal(m.weight.data, 0.1 * torch.ones_like(m.weight.data)))
                assert (torch.equal(m.bias.data, 0 * torch.ones_like(m.weight.data)))

        return model

    def evaluate_meta_learning(self):
        """
        Evaluate meta-trained models by fine-tuning on samples from target and then testing
        :return: average IoU over selections
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

        model_save_dir, logging_save_dir = self.creatEvaluationMetaDirs()
        logger.info("Network:{}".format(self.architecture))

        for meta_method in self.meta_params['methods']:
            pre_trained_model_target_dir = model_save_dir + 'Pre-trained/' + meta_method + '/Target_'
            for selection in self.evaluation_config['selections']:
                selection_dir = self.few_shot_target_dir + 'Selection_' + str(selection) + '/'
                for target in self.evaluation_config['targets']:
                    model_name = self.getPretrainedMetaModelName(meta_method, target)
                    model_pre_train_state_dict_dir = pre_trained_model_target_dir + target + '/' + model_name + '/State_Dict/'
                    model_pretrained = self.load_model_state_dict(state_dict_path=model_pre_train_state_dict_dir +
                                                                                  model_name,epoch=self.state_dict_epoch)
                    if self.evaluation_config['Finetune'] and self.switch_affine:
                        logger.info("Switch Affine to True")
                        model_pretrained = self.swithBatchNormAffine(model_pretrained)
                        print(model_pretrained)

                    for k_shot in self.evaluation_config['k-shot']:

                        logger.info('{} Evaluation\nMeta Method:{}\nSelection: {}\tTarget: {}\tFine-tune Shots: {}'.format(
                                self.lr_method, meta_method, selection, target, k_shot))



                        target_ft_dir, target_test_dir = self.getTargetandFTDir(k_shot=k_shot,
                                                                                selection_dir=selection_dir)

                        finetuneloader, testloader = self.getFTandTestLoader(selection_ft_path=target_ft_dir,
                                                                             selection_test_path=target_test_dir,
                                                                             batchsize_ftset=self.evaluation_config[
                                                                                 'batchsize_ftset'],
                                                                             batchsize_testset=self.evaluation_config[
                                                                                 'batchsize_testset'],
                                                                             dataset=target)

                        experiment_name, _ = self.getExperimentName(k_shot=k_shot,
                                                                    target=target, selection=selection,
                                                                    lr_method='Meta_Learning')
                        model_finetuned_state_dict_dir = model_save_dir + '/Fine-tuned/' + str(k_shot) + \
                                                         '-shot/' + meta_method + '/Target_' + target + '/' + experiment_name

                        writer = SummaryWriter(log_dir=os.getcwd()+'/Logging/Meta_Finetuning/'+self.architecture+'/'+meta_method+
                                                       '/'+experiment_name+'/')

                        logging_ft_prefix = logging_save_dir + 'Fine-tuned/' + str(k_shot) + \
                                                      '-shot/' + meta_method + '/Target_' + target + '/'


                        if self.evaluation_config['Finetune']:
                            print("---Fine-Tuning---")

                            finetune_loss,finetune_iou = self.finetune(finetuneloader=finetuneloader, model=model_pretrained,
                                                                       save_path=model_finetuned_state_dict_dir, logger=logger,
                                                                       writer=writer)
                            result = [finetune_loss,finetune_iou]
                            self.save_result(logging_dir=logging_ft_prefix+'FT_Loss_IoU/',result=result,
                                             experiment_name=experiment_name)


                        if self.evaluation_config['Test_Finetuned']:
                            if self.switch_affine:
                                self.affine=True

                            model = self.load_model_state_dict(state_dict_path=model_finetuned_state_dict_dir)
                            logging_save_target_dir = logging_ft_prefix+'Test_Loss_IoU/'

                            print("---Testing---")

                            test_loss, test_iou, test_acc = self.test(model, testloader)
                            self.affine=False
                            test_result = [test_loss, test_iou, test_acc]
                            self.save_result(result=test_result, logging_dir=logging_save_target_dir,
                                             experiment_name=experiment_name)
                time.sleep(2)
        _,prefix = self.getExperimentName(lr_method=self.lr_method)
        self.results.calc_avg_iou_selections_meta(experiment_name=prefix)

    def evaluate_transfer_learning(self):
        """
           Evaluate supervised-trained models by fine-tuning on samples from target and then testing
           :return: average IoU over selections
         """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

        model_save_dir, logging_save_dir = self.createEvaluationTransferDirs()

        pre_trained_model_target_dir = model_save_dir + 'Pre-trained/Target_'
        pre_train_lr, pre_train_epochs = self.supervised_params['model_lr'], self.supervised_params['epochs']
        logger.info("Network:{}".format(self.architecture))

        for selection in self.evaluation_config['selections']:
            selection_dir = self.few_shot_target_dir + 'Selection_' + str(selection) + '/'
            for target in self.evaluation_config['targets']:
                model_pre_train_state_dict_dir = pre_trained_model_target_dir + target + '/Supervised_Learning_' + \
                                                 pre_train_lr + '_modellr_' + pre_train_epochs + '_epochs_' + target
                model_pretrained = self.load_model_state_dict(state_dict_path=model_pre_train_state_dict_dir +
                                                              self.pretrained_experiment_name_postfix,
                                                              epoch=self.state_dict_epoch)

                if self.evaluation_config['Finetune'] and self.switch_affine:
                    logger.info("Switch Affine to True")
                    for m in model_pretrained.modules():
                        if isinstance(m, nn.Sequential):
                            m[1] = nn.BatchNorm2d(m[1].num_features, affine=True)
                            if self.architecture == 'UNet':
                                m[4] = nn.BatchNorm2d(m[4].num_features, affine=True)
                    print(model_pretrained)
                else:
                    logger.info("Switch Affine to False")

                    print(model_pretrained)
                for k_shot in self.evaluation_config['k-shot']:
                    logger.info(
                        '{} Evaluation\nTransfer Learning\nSelection: {}\tTarget: {}\tFine-tune Shots: {}'.format(
                            self.lr_method, selection, target, k_shot))
                    target_ft_dir, target_test_dir = self.getTargetandFTDir(k_shot=k_shot, selection_dir=selection_dir)

                    finetuneloader, testloader = self.getFTandTestLoader(selection_ft_path=target_ft_dir,
                                                                         selection_test_path=target_test_dir,
                                                                         batchsize_ftset=self.evaluation_config[
                                                                             'batchsize_ftset'],
                                                                         batchsize_testset=self.evaluation_config[
                                                                             'batchsize_testset'],
                                                                         dataset=target)


                    experiment_name, _ = self.getExperimentName(k_shot=k_shot, target=target,selection=selection,
                                                                lr_method='Transfer_Learning')

                    writer = SummaryWriter(log_dir=os.getcwd()+'/Logging/Transfer_Learning/' + self.architecture + '/' +experiment_name + '/')
                    model_finetuned_state_dict_dir = model_save_dir + '/Fine-tuned/' + str(k_shot) + \
                                                     '-shot/' + 'Target_' + target + '/' + experiment_name

                    logging_ft_prefix = logging_save_dir + 'Fine-tuned/' + str(k_shot) + \
                                                        '-shot/Target_' + target + '/'


                    if self.evaluation_config['Finetune']:

                        finetune_loss,finetune_iou = self.finetune(finetuneloader=finetuneloader, model=model_pretrained,
                                                                   save_path=model_finetuned_state_dict_dir, logger=logger,
                                                                   writer=writer)
                        self.save_result(logging_dir=logging_ft_prefix + 'FT_Loss_IoU/',
                                         result=[finetune_loss, finetune_iou],
                                         experiment_name=experiment_name)

                    if self.evaluation_config['Test_Finetuned']:
                        if self.switch_affine:
                            self.affine = True
                        model = self.load_model_state_dict(state_dict_path=model_finetuned_state_dict_dir)
                        logging_save_model_target_dir = logging_ft_prefix+'Test_Loss_IoU/'

                        print("---Testing---")
                        test_loss, test_iou, test_acc = self.test(model, testloader)
                        test_result = [test_loss, test_iou, test_acc]
                        if self.switch_affine:
                            self.affine = False
                        self.save_result(result=test_result, logging_dir=logging_save_model_target_dir,
                                         experiment_name=experiment_name)

            time.sleep(2)

        _, prefix = self.getExperimentName(lr_method=self.lr_method)
        self.results.calc_avg_iou_selections_transfer(experiment_name=prefix)

    def calc_weights(self,labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0,labels.size(0)):
            pos_weight = torch.sum(labels[label_idx]==1)
            neg_weight = torch.sum(labels[label_idx]==0)
            ratio = float(neg_weight.item()/pos_weight.item())
            pos_tensor[label_idx] = ratio*pos_tensor[label_idx]

        return pos_tensor

    def finetune(self, finetuneloader, model, save_path, logger=None,writer=None):
        finetune_loss = 0
        iou_finetune = 0
        acc_finetune = 0
        total_foreground = 0
        finetune_loss_epoch = []
        finetune_iou_epoch = []
        num_samples = 0
        test_iou_best = 0
        best_ft_epoch = 0

        ft_epochs = self.evaluation_config['ft_epochs']
        temp = self.evaluation_config['ft_lr']

        optimizer = optim.Adam(model.parameters(),lr=self.evaluation_config['ft_lr'],weight_decay=self.evaluation_config['optimizer']['weight_decay'])
        model.cuda()

        for e in range(ft_epochs):
            model.train()

            for images, labels in finetuneloader:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                output, _ = model(images)

                iou_temp, intersection_temp,union_temp,acc_temp = self.intersection_over_union(output, labels)
                if self.criterion =='bce':
                    loss = nn.BCELoss()(output, labels)
                else:
                    loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output, labels)

                loss.backward()
                optimizer.step()

                finetune_loss += loss.item() * images.size(0)
                iou_finetune += iou_temp.item()* images.size(0)
                acc_finetune += torch.sum(acc_temp).item()

                total_foreground += torch.sum(labels == 1).item()
                num_samples += images.size(0)

            finetune_loss = finetune_loss / len(finetuneloader.dataset)
            iou_finetune = iou_finetune / len(finetuneloader.dataset)
            acc_finetune = acc_finetune / total_foreground
            logger.info('Epoch:{}//{} \tTrain loss: {:.4f}\tTrain IOU: {:.4f}\t FCA: {:.4f}'.format(e + 1, ft_epochs,
                                                                                                    finetune_loss,iou_finetune,
                                                                                                   acc_finetune))

            if writer !=None:
                writer.add_scalar('Finetune Loss '+os.path.basename(save_path),finetune_loss,e)
                writer.add_scalar('Finetune IoU ' + os.path.basename(save_path),iou_finetune, e)
                writer.add_scalar('Finetune FCA ' + os.path.basename(save_path), acc_finetune, e)

            finetune_loss_epoch.append(finetune_loss)
            finetune_iou_epoch.append(iou_finetune)
            finetune_loss = 0
            acc_finetune = 0

            total_foreground = 0
            num_samples = 0
            iou_finetune = 0
            logger.info('Epoch:{}//{} \tBest test IOU: {:.4f}\t '.format(best_ft_epoch, ft_epochs, test_iou_best))
            if e + 1 == ft_epochs:
                torch.save(model.state_dict(), save_path + '_state_dict.pt')
        self.evaluation_config['ft_lr']=temp
        return finetune_loss_epoch, finetune_iou_epoch

    def test(self, model, testloader):
        iou = 0
        acc = 0

        total_foreground = 0
        test_loss = 0
        model.eval()
        model.cuda()
        for child in model.children():
            if type(child) == nn.Sequential:
                for ii in range(len(child)):
                    if type(child[ii]) == nn.BatchNorm2d:
                        child[ii].track_running_stats = False

        test_start = time.time()
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                output, _ = model(images)
            output=nn.Sigmoid()(output) if self.criterion!='bce' else output
            loss = nn.BCELoss()(output, labels)
            test_loss += loss.item() * images.size(0)
            iou_temp, intersection_temp, union_temp, acc_temp = self.intersection_over_union(output, labels)
            iou += iou_temp.item()* images.size(0)

            acc += torch.sum(acc_temp).item()
            total_foreground += torch.sum(labels == 1).item()
        test_end = time.time()
        test_loss = test_loss / len(testloader.dataset)
        iou = iou / len(testloader.dataset)

        acc = acc / total_foreground
        print('Test Loss: {:.4f} \tTest IOU: {:.4f}\tFCA: {:.4f}\tTest Time: {:.3f} min\n'.format(test_loss, iou, acc,
                                                                                                  (test_end - test_start) / 60))

        return test_loss, iou, acc


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
            iou += torch.mean((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens
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
        return total_iou,torch.sum(intersection_tens).item(),torch.sum(union_tens).item(), foreground_acc

    def save_result(self, logging_dir, result, experiment_name):
        f_pickle = open(logging_dir + experiment_name + '.pickle', 'wb')
        f_csv = open(logging_dir + experiment_name + '.csv', 'w')
        pickle.dump(result, f_pickle)
        df = pd.DataFrame(result)
        df.to_csv(f_csv, header=False, index=False)
        f_pickle.close()
        f_csv.close()


if __name__ == '__main__':

    #torch.manual_seed(123)
    #np.random.seed(123)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    meta_params = {'methods': ['BCE'],
                   'hyperparams': {'meta_lr': '1.0',
                                   'meta_epochs': '700',
                                   'model_lr': '0.001',
                                   'inner_epochs': '30',
                                   'k-shot': '5',
                                   'optimizer': {'weight_decay': '0.0005',
                                                 'momentum': '0.9'}}}

    supervised_params = {'model_lr': '0.001',
                         'epochs': '100'}



    batchsize_testset = {'TNBC': 32,
                         'B39': 32,
                         'ssTEM': 32,
                         'EM': 20,
                         'B5': 32}

    evaluation_config = {'targets': ['TNBC'],
                         'selections': list(range(1, 11)),
                         'k-shot': [3],
                         'batchsize_ftset': 64,
                         'batchsize_testset': batchsize_testset,
                         'ft_lr': 0.0001,
                         'ft_epochs': 20,
                         'optimizer': {'weight_decay': 0.0005,
                                       'momentum': 0.9},
                         'Finetune': True,
                         'Test_Finetuned': True}

    lr_method = 'Meta_Learning'
    evaluation = Evaluation(lr_method=lr_method, evaluation_config=evaluation_config, meta_params=meta_params,loss_function='weightedbce', switch_affine=True,affine=False,
                            state_dict_epoch=300,supervised_params=supervised_params, pretrained_experiment_name_postfix='test',architecture='FCRN',
                            ft_experiment_name_postfix='test')

    if lr_method=='Meta_Learning':
        evaluation.evaluate_meta_learning()
    elif lr_method=='Supervised_Learning':
        evaluation.evaluate_transfer_learning()
    else:
        print("Lr Method is undefined")

from WorkSpace import *
import matplotlib.pyplot as plt
import statistics
import pickle
import pandas as pd
import numpy as np

class Results:
    def __init__(self,lr_method,evaluation_config,meta_methods=None,
                 architecture='FCRN'):
        self.architecture = architecture
        self.lr_method = lr_method
        self.evaluation_config = evaluation_config

        self.prefix = os.getcwd()+'/Logging/{}/{}/'
        self.wrkspace = ManageWorkSpace(datasets=self.evaluation_config['targets'])


    def calc_avg_iou_selections(self,meta_method=None,iou_result_path_prefix=None,
                                experiment_name=None):
        self.wrkspace.create_dir([iou_result_path_prefix + 'CSV/'])


        shots = [str(shot)+' shots' for shot in self.evaluation_config['k-shot']]
        shots = ['']+shots
        iou_list_avg = []
        std_iou_avg = []
        iou_list_avg.append(shots)
        std_iou_avg.append(shots)
        for test in self.evaluation_config['targets']:
            row_iou = []
            std_row_iou = []
            row_iou.append(test)
            std_row_iou.append(test)
            for shot in self.evaluation_config['k-shot']:
                iou_result_path = iou_result_path_prefix+str(shot)+'-shot/'+meta_method+\
                                  '/Target_'+test+'/Test_Loss_IoU/' if meta_method!=None else iou_result_path_prefix+str(shot)+'-shot/Target_'+\
                                                                                test+'/Test_Loss_IoU/'

                row_iou_shot_num = []
                for selection in self.evaluation_config['selections']:
                    experiment_name_selection = experiment_name+'_'+str(shot)+'shot_'+test\
                                                +'_Selection_' + str(selection)
                    try:
                        f = open(iou_result_path + experiment_name_selection + '.pickle', 'rb')
                    except FileNotFoundError:
                        experiment_name_selection = experiment_name + str(shot) + 'shot_' + test \
                                                    + '_Selection_' + str(selection)
                        f = open(iou_result_path + experiment_name_selection + '.pickle', 'rb')
                        test_result = pickle.load(f)
                        iou = test_result[1]
                    else:
                        test_result = pickle.load(f)
                        iou = test_result[1]

                    f.close()

                    row_iou_shot_num.append(iou)
                temp = [elem for elem in row_iou_shot_num]
                std_row_iou.append(round(statistics.stdev(temp), 3))
                row_iou.append(sum(row_iou_shot_num) / len(self.evaluation_config['selections']))
            iou_list_avg.append(row_iou)
            std_iou_avg.append(std_row_iou)

        df = pd.DataFrame(iou_list_avg)
        df_std = pd.DataFrame(std_iou_avg)
        csvFileName = iou_result_path_prefix + 'CSV/'+experiment_name+'_'+meta_method+'.csv' \
                      if meta_method!=None else iou_result_path_prefix + 'CSV/'+experiment_name+'_.csv'

        f = open(csvFileName, 'w')
        with f:
            df.to_csv(f, header=False, index=False)
            df_std.to_csv(f, header=False, index=False)

        f.close()

        return iou_list_avg,std_iou_avg

    def calc_avg_iou_datasets(self,iou, num_datasets):
        iou_temp = []
        for row in iou[1:]:
            temp = row[1:]
            iou_temp.append(temp)
        iou_temp = np.asarray(iou_temp)
        iou_sum = []
        for c in range(iou_temp.shape[1]):
            temp = []
            for r in range(iou_temp.shape[0]):
                temp.append(iou_temp[r][c])
            iou_sum.append(np.sum(temp) / num_datasets)
        return iou_sum

    def calc_avg_iou_selections_meta(self,experiment_name = ''):

        prefix = self.prefix.format(self.wrkspace.map_dict[self.lr_method],self.architecture)
        iou_result_path_prefix = prefix+'Fine-tuned'+'/'

        for i,meta_method in enumerate(self.meta_methods,0):

            iou_avg_selections, std_avg_selections = self.calc_avg_iou_selections(meta_method,iou_result_path_prefix=iou_result_path_prefix,
                                                                     experiment_name=experiment_name)
            print(iou_avg_selections)
            print(std_avg_selections)


    def calc_avg_iou_selections_transfer(self,experiment_name=''):
        prefix = self.prefix.format(self.wrkspace.map_dict[self.lr_method],self.architecture)
        iou_result_path_prefix =prefix+'Fine-tuned/'
        iou_avg_selections, std_avg_selections = self.calc_avg_iou_selections(iou_result_path_prefix=iou_result_path_prefix,
                                                            experiment_name=experiment_name)
        print(iou_avg_selections)
        print(std_avg_selections)




if __name__ == '__main__':

    evaluation_config = {'targets': ['B5'],
                         'selections': [1,2,3,4,5,6,7,8,9,10],
                         'k-shot': [1,3,5,7,10]}


    experiment_name = 'Meta_Learning_finetuned_1.0meta_lr_700meta_epochs_0.001model_lr_30inner_epochs_5shottest_20_ft_epochs_0.0001ft_lr'

    fig_name_prefix = experiment_name+'_'+str(len(evaluation_config['selections']))+'_selections_'
    lr_method = 'Meta_Learning'
    meta_methods = ['BCE']



    results = Results(lr_method=lr_method,evaluation_config=evaluation_config,
                      meta_methods=meta_methods,architecture='UNet')
    if lr_method == 'Meta_Learning':
        results.calc_avg_iou_selections_meta(experiment_name=experiment_name)
    else:
        results.calc_avg_iou_selections_transfer(experiment_name=experiment_name)

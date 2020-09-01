"""
This script runs entire Preprocessing pipeline

"""


import preprocessing
import extractFewShotTargetSelections
import subprocess
import os

root_dir = os.getcwd()
selections = list(range(1,11))
datasets = ['B5','B39','ssTEM','TNBC','EM']
k_shots = [1,3,5,7,10]
target_dir = root_dir+'/Datasets/FewShot/'

ft_crop_steps_dataset = {'B5': {'x': 30, 'y': 30},
                         'B39': {'x': 50, 'y': 26},
                         'EM': {'x': 30, 'y': 30},
                         'TNBC': {'x': 28, 'y': 28},
                         'ssTEM': {'x': 30, 'y': 30}
                         }
source_crop_steps_dataset = {'B5': {'x': 140, 'y': 140},
                             'B39': {'x': 55, 'y': 55},
                             'EM': {'x': 85, 'y': 85},
                             'TNBC': {'x': 20, 'y': 20},
                             'ssTEM': {'x': 32, 'y': 32}
                             }

print("----Starting Data Download-----")
preprocess = preprocessing.Preprocessing(selections=selections, datasets=datasets, k_shots=k_shots, target_dir=target_dir)
subprocess.call(['sh', './downloadUnzipDatasets.sh'])
print("-----Finished Data Download-----")
extractFewShotTargetSelections.extractFewShotTargetSelection()
print("-----Preprocessing Few-Shot Target Selections-----")
preprocess.reprocessFTandTestSamples(crop_steps_dataset=ft_crop_steps_dataset, remove_black_images=True)
print("-----Preprocessing Source Datasets-----")
preprocess.preprocess_Source_Data(crop_steps_dataset=source_crop_steps_dataset, remove_black_images=True)
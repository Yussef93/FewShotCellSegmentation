"""
This script is for generation of barplots
Required to fill in averaged IoU data for each target manually

"""

import numpy as np
import matplotlib.pyplot as plt
import statistics
# load libraries
import pandas as pd


def calc_avg_iou_datasets(iou, num_datasets,res=''):

    iou_temp = np.asarray(iou)
    iou_sum = []
    std_sum = []
    for c in range(iou_temp.shape[1]):
        temp = []
        for r in range(iou_temp.shape[0]):
            temp.append(iou_temp[r][c])
        iou_sum.append(np.sum(temp) / num_datasets)
        std_sum.append(statistics.stdev(temp))

    if res =='average':
        return iou_sum
    else:
        return std_sum





# Create dataframe
raw_data_EM = {'shotnum': ['1-shot', '3-shot', '5-shot', '7-shot', '10-shot'],
               'Transfer_Learning': [],
               'std_TL':[],
               'BCE': [],
               'std_BCE': [],
               'BCE_Entropy': [],
               'std_BCE_Entropy':[],
               'BCE_Distillation': [],
                'std_BCE_Distillation':[],
               'Combined': [],
                'std_Combined':[]
            }

raw_data_TNBC = {'shotnum': ['1-shot', '3-shot', '5-shot', '7-shot', '10-shot'],
               'Transfer_Learning': [],
               'std_TL':[],
               'BCE': [],
               'std_BCE': [],
               'BCE_Entropy': [],
               'std_BCE_Entropy':[],
               'BCE_Distillation': [],
                'std_BCE_Distillation':[],
               'Combined': [],
                'std_Combined':[]
            }

raw_data_ssTEM = {'shotnum': ['1-shot', '3-shot', '5-shot', '7-shot', '10-shot'],
               'Transfer_Learning': [],
               'std_TL':[],
               'BCE': [],
               'std_BCE': [],
               'BCE_Entropy': [],
               'std_BCE_Entropy':[],
               'BCE_Distillation': [],
                'std_BCE_Distillation':[],
               'Combined': [],
                'std_Combined':[]
            }

raw_data_B5 = {'shotnum': ['1-shot', '3-shot', '5-shot', '7-shot', '10-shot'],
               'Transfer_Learning': [],
               'std_TL':[],
               'BCE': [],
               'std_BCE': [],
               'BCE_Entropy': [],
               'std_BCE_Entropy':[],
               'BCE_Distillation': [],
                'std_BCE_Distillation':[],
               'Combined': [],
                'std_Combined':[]
            }

raw_data_B39 = {'shotnum': ['1-shot', '3-shot', '5-shot', '7-shot', '10-shot'],
               'Transfer_Learning': [],
               'std_TL':[],
               'BCE': [],
               'std_BCE': [],
               'BCE_Entropy': [],
               'std_BCE_Entropy':[],
               'BCE_Distillation': [],
                'std_BCE_Distillation':[],
               'Combined': [],
                'std_Combined':[]
            }

dataset = 'EM'
font = 16
df = pd.DataFrame(raw_data_EM, columns = ['shotnum', 'Transfer_Learning','std_TL','BCE','std_BCE',
                                            'BCE_Entropy','std_BCE_Entropy','BCE_Distillation','std_BCE_Distillation',
                                               'Combined','std_Combined'])
print(); print(df)

# Setting the positions and width for the bars
pos = list(range(len(df['BCE'])))
width = 0.125

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with pre_score data
plt.bar(pos, df['Transfer_Learning'], width, alpha=0.7, color='mediumpurple',yerr=df['std_TL'],capsize=5)


# Create a bar with mid_score data,
plt.bar([p + width for p in pos], df['BCE'], width, alpha=0.7, color='royalblue',yerr=df['std_BCE'],capsize=5)

# Create a bar with post_score data,
plt.bar([p + width*2 for p in pos], df['BCE_Entropy'], width, alpha=0.7, color='forestgreen',yerr=df['std_BCE_Entropy'],capsize=5)
plt.bar([p + width*3 for p in pos], df['BCE_Distillation'], width, alpha=0.7, color='orange',yerr=df['std_BCE_Distillation'],capsize=5)
plt.bar([p + width*3.95 for p in pos], df['Combined'], width, alpha=0.7, color='red',yerr=df['std_Combined'],capsize=5)

# Set the y axis label
ax.set_ylabel('mean IoU',size=font)
# Set the chart's title
#ax.set_title(dataset+' FCRN',fontsize=font)

# Set the position of the x ticks
ax.set_xticks([p + 2 * width for p in pos])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font)

# Set the labels for the x ticks
ax.set_xticklabels(df['shotnum'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*6)
plt.ylim([0, 1] )

# Adding the legend and showing the plot
plt.legend(['Transfer Learning', r'ML w. $\mathcal{L}_{BCE}$',r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{ER}$', r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{D}$',r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{ER}+\mathcal{L}_{D}$'], loc='lower right',
           prop={'size':14})
plt.grid()
fig.tight_layout()
plt.savefig('./FCRNFigures/barplot_FCRN_Target_'+dataset+'.png')

iou_methods = [['BCE'],['BCE_Distillation'],['BCE_Entropy'],['Combined'],['TL']]
std_methods = [['BCE'],['BCE_Distillation'],['BCE_Entropy'],['Combined'],['TL']]
iou_BCE = []
iou_BCE.append(raw_data_TNBC['BCE'])
iou_BCE.append(raw_data_B5['BCE'])
iou_BCE.append(raw_data_EM['BCE'])
iou_BCE.append(raw_data_ssTEM['BCE'])
iou_BCE.append(raw_data_B39['BCE'])


iou_BCE_Distillation = []
iou_BCE_Distillation.append(raw_data_TNBC['BCE_Distillation'])
iou_BCE_Distillation.append(raw_data_EM['BCE_Distillation'])
iou_BCE_Distillation.append(raw_data_ssTEM['BCE_Distillation'])
iou_BCE_Distillation.append(raw_data_B5['BCE_Distillation'])
iou_BCE_Distillation.append(raw_data_B39['BCE_Distillation'])

iou_BCE_Entropy = []
iou_BCE_Entropy.append(raw_data_TNBC['BCE_Entropy'])
iou_BCE_Entropy.append(raw_data_EM['BCE_Entropy'])
iou_BCE_Entropy.append(raw_data_ssTEM['BCE_Entropy'])
iou_BCE_Entropy.append(raw_data_B5['BCE_Entropy'])
iou_BCE_Entropy.append(raw_data_B39['BCE_Entropy'])


iou_Combined = []
iou_Combined.append(raw_data_TNBC['Combined'])
iou_Combined.append(raw_data_EM['Combined'])
iou_Combined.append(raw_data_ssTEM['Combined'])
iou_Combined.append(raw_data_B5['Combined'])
iou_Combined.append(raw_data_B39['Combined'])


iou_Transfer = []
iou_Transfer.append(raw_data_TNBC['Transfer_Learning'])
iou_Transfer.append(raw_data_EM['Transfer_Learning'])
iou_Transfer.append(raw_data_ssTEM['Transfer_Learning'])
iou_Transfer.append(raw_data_B5['Transfer_Learning'])
iou_Transfer.append(raw_data_B39['Transfer_Learning'])

iou_methods[1].append(calc_avg_iou_datasets(iou=iou_BCE,num_datasets=5,res='average'))
iou_methods[3].append(calc_avg_iou_datasets(iou=iou_BCE_Distillation,num_datasets=5,res='average'))
iou_methods[2].append(calc_avg_iou_datasets(iou=iou_BCE_Entropy,num_datasets=5,res='average'))
iou_methods[4].append(calc_avg_iou_datasets(iou=iou_Combined,num_datasets=5,res='average'))
iou_methods[0].append(calc_avg_iou_datasets(iou=iou_Transfer,num_datasets=5,res='average'))

std_methods[1].append(calc_avg_iou_datasets(iou=iou_BCE,num_datasets=5))
std_methods[3].append(calc_avg_iou_datasets(iou=iou_BCE_Distillation,num_datasets=5))
std_methods[2].append(calc_avg_iou_datasets(iou=iou_BCE_Entropy,num_datasets=5))
std_methods[4].append(calc_avg_iou_datasets(iou=iou_Combined,num_datasets=5))
std_methods[0].append(calc_avg_iou_datasets(iou=iou_Transfer,num_datasets=5))


map_fig_title = {'BCE':r'ML w. $\mathcal{L}_{BCE}$',
                     'BCE_Entropy':r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{ER}$',
                     'BCE_Distillation':r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{D}$',
                     'Combined':r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{ER}+\mathcal{L}_{D}$',
                     'TL':'Transfer Learning',
                     'RandomInit':'Random Initialization'}
legend_methods = map_fig_title

print(iou_methods)
markers = ['-x', '-d', '-*', '-o', '-s']
color = ['mediumpurple','royalblue','forestgreen','orange','red']

fig, ax = plt.subplots(figsize=(10,5))
temp=[]
place = 0

for i in range(len(iou_methods)):
    plt.bar([p + width*place for p in pos], iou_methods[i][1], width, alpha=0.7, color=color[i],capsize=5)
    place = place+1
    temp.append(legend_methods[iou_methods[i][0]])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font)

# Set the labels for the x ticks
ax.set_xticklabels(['1-shot', '3-shot', '5-shot', '7-shot', '10-shot'])
ax.set_xticks([p + 2 * width for p in pos])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*6)

legend_methods = temp
plt.title('Average across Databases (FCRN) ', fontsize=font)
plt.ylabel('mean IoU', fontsize=font)

plt.ylim(0.5, 0.75)
plt.grid()
plt.legend(['Transfer Learning', r'ML w. $\mathcal{L}_{BCE}$',r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{ER}$', r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{D}$',r'ML w. $\mathcal{L}_{BCE}+\mathcal{L}_{ER}+\mathcal{L}_{D}$'], loc='lower right',
           prop={'size':14})
fig.tight_layout()
plt.savefig('./FCRNFigures/AveragedIoUmethodsFCRN.png')
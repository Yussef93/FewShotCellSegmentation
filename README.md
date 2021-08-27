# FewShotCellSegmentation

Automatic cell segmentation in microscopy images works well with the support of deep neural networks trained with full supervision. Collecting and annotating images, though, is not a sustainable solution for every new microscopy database and cell type. Instead, we assume that we can access a plethora of annotated image data sets from different domains (sources) and a limited number of annotated image data sets from the domain of interest (target), where each domain denotes not only different image appearance but also a different type of cell segmentation problem. We pose this problem as meta-learning where the goal is to learn a generic and adaptable few-shot learning model from the available source domain data sets and cell segmentation tasks. The model can be afterwards fine-tuned on the few annotated images of the target domain that contains different image appearance and different cell type. In our meta-learning training, we propose the combination of three objective functions to segment the cells, move the segmentation results away from the classification boundary using cross-domain tasks, and learn an invariant representation between tasks of the source domains. Our experiments on five public databases show promising results from 1- to 10-shot meta-learning using standard segmentation neural network architectures.

Link to full paper https://arxiv.org/abs/2007.01671
## Algorithm

![Screenshot 2020-07-06 at 13 12 29](https://user-images.githubusercontent.com/57146761/86587695-676fe580-bf8a-11ea-92c8-b11ff30dd519.png)


## Results
![Screenshot 2020-07-06 at 13 00 37](https://user-images.githubusercontent.com/57146761/86587341-b0736a00-bf89-11ea-802e-abb537784daa.png)

![Screenshot 2020-07-06 at 13 06 23](https://user-images.githubusercontent.com/57146761/86587480-f4ff0580-bf89-11ea-99bb-ef4c5628b8cf.png)

![Screenshot 2020-07-06 at 13 06 41](https://user-images.githubusercontent.com/57146761/86587456-ea447080-bf89-11ea-807d-8b1591d10003.png)
## Code
1- Install necessary python modules in requirements.txt

2- Run run_preprocessing.py i.e. python run_preprocessing.py to download the datasets and preprocess them, in addition to extracting and preprocessing my 10 random selections.

3- Instructions to run training and evaluation are available with examples in Learning_main.py and Evaluation_main.py

## Pre-trained Models
The Pre-trained models can be downloaded from this link https://cloudstore.uni-ulm.de/s/YqD6or4DLyjF7ry 

## License
This project is licensed under the MIT license - see the License.md file for details

## Cite
To cite this repository, please use the following citation:


```
@inproceedings{DBLP:conf/pkdd/DawoudHCB20,
  author    = {Youssef Dawoud and
               Julia Hornauer and
               Gustavo Carneiro and
               Vasileios Belagiannis},
  title     = {Few-Shot Microscopy Image Cell Segmentation},
  booktitle = {{ECML/PKDD} {(5)}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12461},
  pages     = {139--154},
  publisher = {Springer},
  year      = {2020}
} 
```

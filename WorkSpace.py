import os
import shutil

class ManageWorkSpace:

    def __init__(self,root_dir=None,datasets=None):
        self.root_dir = root_dir if root_dir else os.getcwd()
        self.default_dir_dict = {'datasets_dir':['/Datasets/FewShot/Source/','/Datasets/FewShot/Target/'],
                                 'Logging':['/Logging/Meta-models/','/Logging/Supervised-models/'],
                                 'models':['/models/Meta-models/','/models/Supervised-models/']}


        self.map_dict = {'Meta_Learning':'Meta-models',
                         'Supervised_Learning':'Supervised-models',
                         'RandomInit':'Random-models'}

        if os.path.basename(self.root_dir)=='FewShotCellSegmentation':
            self.create_default()

    def remove_dir(self,dirs:list):
        if len(dirs)==1:
            if os.path.exists(dirs[0]):
                shutil.rmtree(dirs[0])
        else:
            for dir in dirs:
                if os.path.exists(dir):
                    shutil.rmtree(dir)

    def create_dir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    #Create Default Workspace Directories
    def create_default(self):
        for dirs in self.default_dir_dict.keys():
            for dir in self.default_dir_dict[dirs]:
                self.create_dir(self.root_dir+"/"+dir)

if __name__ == '__main__':
    datasets = ['B5','B39','ssTEM','EM','TNBC']
    wrkSpace = ManageWorkSpace(datasets=datasets)

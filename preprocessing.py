"""
This Python class is specified for Pre-processing microscopy image datasets
- Preprocessing --> Class
- Source datasets are preprocessed using preprocess_Source_Data()
- Target dataset selections are selected and preprocessed using preprocess_Target_Data()
- To preprocess our 10 selections use reprocessFTandTestSamples()

"""

from collections import Counter
from WorkSpace import *
import numpy as np
import re
import ntpath
from PIL import  Image,ImageChops

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_slices(image, slice_size, steps_x,steps_y, remove_black_images, threshold):

    slices = []
    image_array = np.array(image)
    for x in range(0, image_array.shape[0], steps_x):
        for y in range(0, image_array.shape[1], steps_y):
            if x+slice_size <= image_array.shape[0] and y+slice_size <= image_array.shape[1]:
                img = image_array[x:x+slice_size, y:y+slice_size]
                if np.count_nonzero(img) < threshold and remove_black_images==True:
                    img = None
                slices.append(img)
    return slices

class Preprocessing():

    def __init__(self, selections,datasets, target_dir,
                 k_shots=None, crop=True,source_dir=os.getcwd()+'/Datasets/Raw/'):

        self.root_dir = source_dir
        self.target_dir = target_dir
        self.datasets = datasets
        self.selections = selections
        self.k_shots = k_shots
        self.crop = crop
        self.wrkspace = ManageWorkSpace(datasets=datasets)




    def createFewShotTargetDirs(self):
        # Create Target Dataset Directories under
        # FewShotCellSegmentation/Dataset/FewShot/Target/

        for selection in self.selections:
            fewshot_target_dir = self.target_dir +'Target/'
            prefix = fewshot_target_dir+'Selection_' + str(selection) + '/'
            fine_tune_dir,test_dir = prefix + 'FinetuneSamples/',prefix + 'TestSamples/'


            for k_shot in self.k_shots:
                for dataset in self.datasets:
                    dirs = [fine_tune_dir+str(k_shot)+'-shot/'+dataset +'/Image/',
                            fine_tune_dir+str(k_shot)+'-shot/'+dataset +'/Groundtruth/',
                            test_dir+str(k_shot)+'-shot/'+dataset +'/Image/',
                            test_dir+str(k_shot)+'-shot/'+dataset +'/Groundtruth/']
                    preprocess_dirs = [fine_tune_dir+str(k_shot)+'-shot/'+'preprocessed/' + dataset+'/Image/',
                                       fine_tune_dir+str(k_shot)+'-shot/'+ 'preprocessed/' + dataset+'/Groundtruth/']
                    self.wrkspace.remove_dir(dirs+preprocess_dirs)
                    self.wrkspace.create_dir(dirs+preprocess_dirs)

    def createFewShotSourceDirs(self):
        # Create Source Dataset Directories under
        # FewShotCellSegmentation/Dataset/FewShot/Source/

        fewshot_source_dir = self.target_dir + 'Source/'
        for dataset in self.datasets:
            dirs = [fewshot_source_dir + dataset+ '/Image/',
                    fewshot_source_dir + dataset+'/Groundtruth/',
                    ]
            self.wrkspace.remove_dir(dirs)
            self.wrkspace.create_dir(dirs)


    def getRawImagesAndGroundtruth(self,dataset):
        # Get Raw Dataset Images FewShotCellSegmentation/Dataset/Raw/

        image_files = []
        ground_truth_files = []
        ground_truth_prefix = self.root_dir+dataset+'/Groundtruth/'
        image_prefix = self.root_dir+dataset+'/Image/'
        if dataset == 'TNBC':
            ground_truth_folders = sorted(
                [folder + '/' for folder in os.listdir(ground_truth_prefix) if folder[0] == 'G'])
            image_folders = sorted(
                [folder + '/' for folder in os.listdir(image_prefix) if folder[0] == 'S'])

            for folder in ground_truth_folders:
                ground_truth_files += sorted([ground_truth_prefix + folder + f for f in
                                              os.listdir(ground_truth_prefix + folder) if
                                              f[0] != '.'])
            for folder in image_folders:
                image_files += sorted(
                    [image_prefix + folder + f for f in os.listdir(image_prefix + folder) if
                     f[0] != '.'])
        else:
            ground_truth_files = sorted(
                [ground_truth_prefix + f for f in os.listdir(ground_truth_prefix) if f[0] != '.'])
            image_files = sorted([image_prefix + f for f in os.listdir(image_prefix) if f[0] != '.'])[:len(ground_truth_files)]

        return image_files,ground_truth_files

    def getTestImagesandGroundtruth(self,dataset,fine_tune_dir):
        #Get Test Image samples which are not in Few-Shot samples

        image_files = []
        ground_truth_files = []
        ground_truth_prefix = self.root_dir + dataset + '/Groundtruth/'
        image_prefix = self.root_dir + dataset + '/Image/'
        if dataset == 'TNBC':
            ground_truth_folders = sorted(
                [folder + '/' for folder in os.listdir(ground_truth_prefix) if folder[0] == 'G'])
            image_folders = sorted(
                [folder + '/' for folder in os.listdir(image_prefix) if folder[0] == 'S'])

            for folder in image_folders:
                image_files += sorted([image_prefix + folder + f for f in os.listdir(image_prefix + folder)
                                       if f[0] != '.' and f not in os.listdir(fine_tune_dir+dataset+'/Image/')])

            for folder in ground_truth_folders:
                ground_truth_files += sorted([ground_truth_prefix + folder + f for f in
                                              os.listdir(ground_truth_prefix + folder) if
                                              f[0] != '.' and f not in os.listdir(fine_tune_dir+dataset+'/Groundtruth/')])
        else:
            ground_truth_files = sorted([ground_truth_prefix + f for f in os.listdir(ground_truth_prefix) if f[0] != '.' and
                 f not in os.listdir(fine_tune_dir+dataset+'/Groundtruth/')])

            image_files = sorted([image_prefix + f for f in os.listdir(image_prefix) if f[0] != '.' and
                 f not in os.listdir(fine_tune_dir+dataset+'/Image/')])[:
                len(ground_truth_files)]

        return image_files, ground_truth_files

    def savefiles(self,image_files,ground_truth_files,save_dir):
        for file in image_files:
            image = Image.open(file)
            base_name = os.path.basename(file)
            image.save(save_dir + '/Image/' + base_name)

        for file in ground_truth_files:
            image = Image.open(file)
            base_name = os.path.basename(file)
            image.save(save_dir + '/Groundtruth/' + base_name)

    def selectKRandomShots(self,ground_truth_files,image_files,dataset,shot,
                           save_dir):
        # select K-shots from a dataset for fine-tuning
        isFileRepeated = True
        while isFileRepeated:
            ground_truth_temp = np.random.choice(ground_truth_files, shot)
            ground_truth_temp = sorted(ground_truth_temp)
            counter = Counter(ground_truth_temp)
            cnt_bol = []
            for key in counter:
                if counter[key] > 1:
                    cnt_bol.append(True)
            if len(cnt_bol) == 0:
                isFileRepeated = False
                ground_truth_files = ground_truth_temp

        ground_truth_files  = sorted(ground_truth_files)
        image_files = self.getKshotImageFiles(ground_truth_files,image_files,dataset)
        self.savefiles(image_files=image_files,ground_truth_files=ground_truth_files,save_dir=save_dir)

        return image_files,ground_truth_files

    def getKshotImageFiles(self,ground_truth_files,image_files,dataset):

        if dataset == 'ssTEM' or dataset == 'B39':
            fileCode = re.compile(r'\d+')
            base_name = []
            temp = []
            for ground_truth_file in ground_truth_files:
                base_name.append(path_leaf(ground_truth_file))
            for i in range(0, len(ground_truth_files)):
                for image_file in image_files:
                    if len(fileCode.findall(os.path.basename(image_file))) == 1:
                        if fileCode.findall(os.path.basename(image_file)) == fileCode.findall(base_name[i]):
                            temp.append(image_file)
                    elif fileCode.findall(os.path.basename(image_file)) == fileCode.findall(base_name[i]):
                        temp.append(image_file)
        else:
            base_name = []
            temp = []
            for ground_truth_file in ground_truth_files:
                base_name.append(os.path.basename(ground_truth_file))
            for image_file in image_files:
                if os.path.basename(image_file) in base_name:
                    temp.append(image_file)

        image_files = sorted(temp)
        assert (len(ground_truth_files) == len(image_files))
        return image_files

    def preprocess_Groundtruth(self,ground_truth_files,dataset, size=256, steps=None,
                               remove_black_images=False, threshold=360,crop=True):


        preprocessed_ground_truth = []
        for file in ground_truth_files:
            ground_truth = Image.open(file)
            if dataset == 'ssTEM' and file.find('labels') != -1:
                ground_truth = ImageChops.invert(ground_truth)
                ground_truth = np.array(ground_truth)
                ground_truth[ground_truth >= 0.5] = 255
                ground_truth[ground_truth < 0.5] = 0
            elif dataset == 'B39':
                ground_truth = np.array(ground_truth)
                ground_truth = np.matmul(ground_truth, [0.2989, 0.5870, 0.1140, 0])
                ground_truth[ground_truth > 0] = 255
                ground_truth = ground_truth.astype(np.uint8)

            else:
                ground_truth = np.array(ground_truth)
                ground_truth[ground_truth >= 0.5] = 255
                ground_truth[ground_truth < 0.5] = 0
            if crop:
                preprocessed_ground_truth += get_slices(ground_truth, size, steps_x=steps['x'], steps_y=steps['y'],
                                                        remove_black_images=remove_black_images,
                                                        threshold=threshold)
            else:
                preprocessed_ground_truth += [ground_truth]

        return preprocessed_ground_truth

    def preprocess_Images(self,image_files,
                               size=256, steps=None, remove_black_images=False, threshold=360,crop=True):
        preprocessed_images = []
        if crop:
            for file in image_files:
                image = Image.open(file)
                image = image.convert('L')
                preprocessed_images += get_slices(image, size, steps_x=steps['x'], steps_y=steps['y'],
                                                  remove_black_images=remove_black_images, threshold=threshold)
        else:
            for file in image_files:
                image = Image.open(file)
                image = image.convert('L')
                image = np.array(image)
                preprocessed_images+=[image]

        return preprocessed_images

    def preprocess(self,ground_truth_files,image_files,dataset,
                   crop_window_size=256, steps=None,remove_black_images=False,crop=None):
        return self.preprocess_Images(image_files,crop_window_size,steps=steps,remove_black_images=remove_black_images,crop=crop),\
               self.preprocess_Groundtruth(ground_truth_files,dataset,crop_window_size,steps=steps,
                                           remove_black_images=remove_black_images,crop=crop)


    def save_preprocessed_data(self,preprocessed_ground_truth,preprocessed_images,
                               save_dir,target=None):

        img_count = 0
        for i, sample in enumerate(preprocessed_ground_truth):
            if sample is not None:
                img_count += 1
                img = Image.fromarray(sample)
                img = img.convert('L')
                img.save(save_dir['groundtruth']+'ground_truth{}.png'.format(img_count))

        img_count = 0
        for i, sample in enumerate(preprocessed_images):
            if preprocessed_ground_truth[i] is not None:
                img_count += 1
                img = Image.fromarray(sample)
                img.save(save_dir['image']+'image{}.png'.format(img_count))


    def preprocess_Target_Data(self,crop_window_size=256, crop_steps_dataset=None, remove_black_images=False,
                                threshold=360, test_samples=None):

        # main function for extracting few-shot/Test samples selections and
        # pre-processing Target Data in the leave-one-dataset-out cross-validaton

        self.createFewShotTargetDirs()
        crop = self.crop
        for selection in self.selections:

            print("Processing Selection {}".format(selection))
            fewshot_target_dir = self.target_dir + 'Target/'
            prefix = fewshot_target_dir + 'Selection_' + str(selection) + '/'

            for shot in self.k_shots:
                fine_tune_dir = prefix + 'FinetuneSamples/'+str(shot)+'-shot/'
                test_dir = prefix + 'TestSamples/'+str(shot)+'-shot/'
                for dataset in self.datasets:
                    save_ft_dir = {'image':fine_tune_dir + 'preprocessed/' +dataset + '/Image/',
                                'groundtruth':fine_tune_dir + 'preprocessed/' +dataset + '/Groundtruth/'}

                    save_test_dir = {'image': test_dir + dataset + '/Image/',
                                   'groundtruth': test_dir + dataset + '/Groundtruth/'}
                    image_files,ground_truth_files = self.getRawImagesAndGroundtruth(dataset)
                    image_files,ground_truth_files = self.selectKRandomShots(ground_truth_files,image_files,dataset,shot,
                                                                             save_dir=fine_tune_dir+dataset)
                    preprocessed_images,preprocessed_ground_truth = self.preprocess(ground_truth_files,image_files,dataset,crop=crop,
                                                                                    crop_window_size=crop_window_size,steps=crop_steps_dataset[dataset],
                                                                                    remove_black_images=remove_black_images)
                    self.save_preprocessed_data(preprocessed_ground_truth=preprocessed_ground_truth,preprocessed_images=preprocessed_images,
                                                save_dir=save_ft_dir)

                    test_images,test_groundtruth = self.getTestImagesandGroundtruth(dataset,fine_tune_dir)
                    crop = False
                    save_dir = {'image':test_dir + dataset + '/Image/','groundtruth':test_dir + dataset + '/Groundtruth/'}
                    test_images, test_groundtruth = self.preprocess(test_groundtruth,test_images,dataset,crop=crop)
                    self.save_preprocessed_data(test_groundtruth,test_images,save_dir=save_test_dir)
                    crop = True

            print("Processed Selection {}".format(selection))

    def preprocess_Source_Data(self, crop_window_size=256, crop_steps_dataset=None, remove_black_images=False,
                               threshold=360):

        # main function for pre-processing Source Data combination in the leave-one-dataset-out cross-validaiton

        self.createFewShotSourceDirs()
        crop = self.crop


        fewshot_source_dir = self.target_dir + 'Source/'
        for dataset in self.datasets:
            prefix = fewshot_source_dir + dataset + '/'
            save_dir = {'image':prefix+'Image/','groundtruth':prefix+'Groundtruth/'}
            self.wrkspace.create_dir([save_dir['image'],save_dir['groundtruth']])
            image_files, ground_truth_files = self.getRawImagesAndGroundtruth(dataset)
            preprocessed_images, preprocessed_ground_truth = self.preprocess(ground_truth_files, image_files,crop_window_size=crop_window_size,
                                                                             remove_black_images=remove_black_images,
                                                                             steps=crop_steps_dataset[dataset],
                                                                             dataset=dataset, crop=crop)
            self.save_preprocessed_data(preprocessed_ground_truth=preprocessed_ground_truth,
                                        preprocessed_images=preprocessed_images,
                                        save_dir=save_dir)

            print("Processed Source: ",dataset)



    def reprocessFTandTestSamples(self,crop_window_size=256, crop_steps_dataset=None,
                                              remove_black_images=False):

        # Preprocessing the Fine-tuning shots and test samples of the 10 selections
        # used in the Paper's experiments

        crop = self.crop
        for selection in self.selections:

            print("Processing Selection {}".format(selection))
            fewshot_target_dir = self.target_dir + 'Target/'
            prefix = fewshot_target_dir + 'Selection_' + str(selection) + '/'
            remove = True
            for shot in self.k_shots:
                fine_tune_dir = prefix + 'FinetuneSamples/' + str(shot) + '-shot/'
                test_dir = prefix + 'TestSamples/' + str(shot) + '-shot/'
                if remove:
                    self.wrkspace.remove_dir([fine_tune_dir+'preprocessed/'])

                for dataset in self.datasets:
                    preprocess_dirs = [fine_tune_dir + 'preprocessed/' + dataset + '/Image/',
                                       fine_tune_dir + 'preprocessed/' + dataset + '/Groundtruth/']

                    self.wrkspace.create_dir(preprocess_dirs)
                    save_ft_dir = {'image':fine_tune_dir + 'preprocessed/' +dataset + '/Image/',
                                'groundtruth':fine_tune_dir + 'preprocessed/' +dataset + '/Groundtruth/'}

                    save_test_dir = {'image': test_dir + dataset + '/Image/',
                                   'groundtruth': test_dir + dataset + '/Groundtruth/'}


                    fewShot_image_files= sorted([fine_tune_dir+dataset+'/Image/'+f for f in os.listdir(fine_tune_dir+dataset+'/Image/')])
                    fewShot_ground_truth_files = sorted([fine_tune_dir+dataset+'/Groundtruth/'+f for f in os.listdir(fine_tune_dir+dataset+'/Groundtruth/')])

                    test_image_files = sorted([test_dir + dataset + '/Image/' + f for f in
                                                  os.listdir(test_dir + dataset + '/Image/')])
                    test_ground_truth_files = sorted([test_dir + dataset + '/Groundtruth/' + f for f in
                                                         os.listdir(test_dir + dataset + '/Groundtruth/')])


                    preprocessed_fewshot_images,preprocessed_fewShot_ground_truth = self.preprocess(fewShot_ground_truth_files,fewShot_image_files,dataset,steps=crop_steps_dataset[dataset],
                                                                                    remove_black_images=remove_black_images,crop=crop)

                    preprocessed_test_images, preprocessed_test_ground_truth = self.preprocess(test_ground_truth_files, test_image_files, dataset, steps=crop_steps_dataset[dataset],
                        remove_black_images=remove_black_images, crop=False)

                    assert(len(preprocessed_fewshot_images)==len(preprocessed_fewShot_ground_truth))
                    assert (len(preprocessed_test_images) == len(preprocessed_test_ground_truth))
                    self.save_preprocessed_data(preprocessed_ground_truth=preprocessed_fewShot_ground_truth,preprocessed_images=preprocessed_fewshot_images,
                                                save_dir=save_ft_dir)


                    for test_image_file,test_ground_truth_file in zip(test_image_files,test_ground_truth_files):
                        os.remove(test_image_file)
                        os.remove(test_ground_truth_file)

                    self.save_preprocessed_data(preprocessed_ground_truth=preprocessed_test_ground_truth,
                                                preprocessed_images=preprocessed_test_images,
                                                save_dir=save_test_dir)



    def preprocess_Data(self,crop_window_size=256, crop_steps_dataset=None, remove_black_images=False,
                                threshold=360):

        self.preprocess_Target_Data(crop_window_size=crop_window_size, crop_steps_dataset=crop_steps_dataset)
        self.preprocess_Source_Data(crop_window_size=crop_window_size, crop_steps_dataset=crop_steps_dataset)





if __name__ == '__main__':
    root_dir = os.getcwd()
    selections = list(range(1,11))
    datasets = ['B5','B39','ssTEM','TNBC','EM']
    k_shots = [1,3,5,7,10]
    target_dir = root_dir+'/Datasets/FewShot/'

    ft_crop_steps_dataset = {'B5':{'x':30,'y':30},
                          'B39':{'x':50,'y':26},
                          'EM':{'x':30,'y':30},
                          'TNBC':{'x':28,'y':28},
                          'ssTEM':{'x':30,'y':30}
                          }
    source_crop_steps_dataset = {'B5': {'x':140,'y':140},
                          'B39': {'x': 55,'y':55},
                          'EM': {'x':85,'y':85},
                          'TNBC': {'x':20,'y':20},
                          'ssTEM': {'x':32,'y':32}
                          }

    preprocessing = Preprocessing(selections=selections, datasets=datasets, k_shots=k_shots, target_dir=target_dir)
    preprocessing.reprocessFTandTestSamples(crop_steps_dataset=ft_crop_steps_dataset, remove_black_images=True)
    preprocessing.preprocess_Source_Data(crop_steps_dataset=source_crop_steps_dataset, remove_black_images=True)




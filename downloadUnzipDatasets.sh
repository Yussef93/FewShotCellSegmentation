#!/bin/bash

trap 'rm -r ./Downloads/' EXIT
mkdir ./Datasets/Downloads/ && cd ./Datasets/Downloads
mkdir EM_images/ && mkdir EM_groundtruth/

echo '---Downloading B5 Images'
curl -o B5_images.zip https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
unzip B5_images.zip
mv ./BBBC005_v1_images/*.TIF ../Raw/B5/Image/

echo '---Downloading B5 Groundtruth'
curl -o B5_groundtruth.zip https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip
echo 'Unzipping...'
unzip B5_groundtruth.zip
mv ./BBBC005_v1_ground_truth/*.TIF ../Raw/B5/Groundtruth/

echo '---Downloading B39 Images'
curl -o B39_images.zip https://data.broadinstitute.org/bbbc/BBBC039/images.zip
echo 'Unzipping...'
unzip B39_images.zip
mv ./images/*.tif ../Raw/B39/Image/

ehco '---Downloading B39 Groundtruth'
curl -o B39_groundtruth.zip https://data.broadinstitute.org/bbbc/BBBC039/masks.zip
echo 'Unzipping...'
unzip B39_groundtruth.zip
mv ./masks/*.png ../Raw/B39/Groundtruth/

echo '---Downloading TNBC'
curl -o TNBC.zip https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip?download=1
unzip TNBC.zip
mv ./TNBC_NucleiSegmentation/Slide_* ../Raw/TNBC/Image/
mv ./TNBC_NucleiSegmentation/GT_* ../Raw/TNBC/Groundtruth/

echo '---Downloading ssTEM'
curl -o ssTEM.zip -L https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/master.zip
unzip ssTEM.zip
mv ./groundtruth-drosophila-vnc-master/stack1/raw/* ../Raw/ssTEM/Image/
mv ./groundtruth-drosophila-vnc-master/stack1/mitochondria/* ../Raw/ssTEM/Groundtruth/

echo '---Downloading EM'
cd ./EM_images
curl -o images  https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif
tiffsplit images
rm  images
mv *.tif ../../Raw/EM/Image/

cd ../EM_groundtruth
curl -o groundtruth https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training_groundtruth.tif
tiffsplit groundtruth
rm groundtruth
mv *.tif ../../Raw/EM/Groundtruth/
cd ..

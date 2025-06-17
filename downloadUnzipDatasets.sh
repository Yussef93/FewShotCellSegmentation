#!/bin/bash

set -euo pipefail

mkdir -p ./Datasets/Downloads
cd ./Datasets/Downloads

mkdir -p EM_images EM_groundtruth
mkdir -p ../Raw/{B5,EM,ssTEM,TNBC,B39}/{Image,Groundtruth}


echo '---Downloading B5 Images'
curl -o B5_images.zip https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip
unzip B5_images.zip
rm -rf B5_images.zip
mv ./BBBC005_v1_images/*.TIF ../Raw/B5/Image/

echo '---Downloading B5 Groundtruth'
curl -o B5_groundtruth.zip https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip
unzip B5_groundtruth.zip
rm -rf B5_groundtruth.zip
mv ./BBBC005_v1_ground_truth/*.TIF ../Raw/B5/Groundtruth/

echo '---Downloading B39 Images'
curl -o B39_images.zip https://data.broadinstitute.org/bbbc/BBBC039/images.zip
unzip B39_images.zip
rm -rf B39_images.zip
mv ./images/*.tif ../Raw/B39/Image/

echo '---Downloading B39 Groundtruth'
curl -o B39_groundtruth.zip https://data.broadinstitute.org/bbbc/BBBC039/masks.zip
unzip B39_groundtruth.zip
mv ./masks/*.png ../Raw/B39/Groundtruth/

echo '---Downloading TNBC'
curl -L -o TNBC.zip "https://zenodo.org/records/1175282/files/TNBC_NucleiSegmentation.zip?download=1"
unzip TNBC.zip
mv ./TNBC_NucleiSegmentation/Slide_* ../Raw/TNBC/Image/
mv ./TNBC_NucleiSegmentation/GT_* ../Raw/TNBC/Groundtruth/

echo '---Downloading ssTEM'
curl -o ssTEM.zip -L https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/master.zip
unzip ssTEM.zip
mv ./groundtruth-drosophila-vnc-master/stack1/raw/* ../Raw/ssTEM/Image/
mv ./groundtruth-drosophila-vnc-master/stack1/mitochondria/* ../Raw/ssTEM/Groundtruth/


echo '--- Downloading EM'
cd EM_images
curl -L -o images  https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif
tiffsplit images
rm images
mv *.tif ../Raw/EM/Image/
cd ../EM_groundtruth
curl -o groundtruth https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training_groundtruth.tif
tiffsplit groundtruth
rm groundtruth
mv *.tif ../Raw/EM/Groundtruth/
cd ../..

echo 'All datasets downloaded and extracted successfully.'
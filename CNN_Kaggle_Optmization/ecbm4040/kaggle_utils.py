#!/usr/bin/env python
# ECBM E4040 Neural Networks and Deep Learning
# This is a utility function to help you download the dataset and preprocess the data we use for this homework.
# requires several modules: _pickle, tarfile, glob. If you don't have them, search the web on how to install them.
# You are free to change the code as you like.

# Import modules. If you don't have them, try `pip install xx` or `conda
# install xx` in your console.
import _pickle as pickle
import os
import tarfile
import zipfile
import glob
import urllib.request as url
import numpy as np
import scipy.misc
from PIL import Image


def download_train_data():
    
    if not os.path.exists('./k-data/kaggle_train_128.zip'):
        os.mkdir('./k-data')
        print('Start downloading data...')
        url.urlretrieve("https://ecbm4040.bitbucket.io/kaggle_train_128.zip","./k-data/kaggle_train_128.zip")
        print('Download complete.')
    else:
        if os.path.exists('./k-data/kaggle_train_128.zip'):
            print('Kaggle package already exists.')


def load_data(mode='all'):
    
    if not os.path.exists('./k-data/kaggle_train_128.zip'):
        download_train_data()
    else:
        print('./k-data/kaggle_train_128.zip already exists. Begin extracting...')
        
    root_dir = os.getcwd()
    os.chdir('./k-data')     
    if not os.path.exists('./train_128'):
        os.system("unzip kaggle_train_128.zip")
    else:
        print("exists already!")
    
    train_data = []
    train_label = []

    
    extension = ".png"
    path_to_image_folder = "./train_128/"
    num_train_samples_per_class = 3000 
    num_classes = 5
    train_img_names = []
    train_labels = []
    
    for _class in range(num_classes):
        path = path_to_image_folder + str(_class) + "/"
        print("Current status:")
        print(path)
        train_img_names += [path + str(idx) + extension for idx in range(_class*num_train_samples_per_class,(_class+1)*num_train_samples_per_class)]
        train_labels += [str(_class)]*num_train_samples_per_class

        
    #folder="0/"
    #dir_name = './train_128/'+str(folder)
    #os.chdir(dir_name)
    train_images = []
    for img_name in train_img_names:
        im = Image.open(img_name)
        im = im.resize((96,96),Image.ANTIALIAS)
        train_images.append(np.array(im))
        
    os.chdir(root_dir)
    X_train = np.asarray(train_images)
    
    return X_train,np.asarray(train_labels)
    
    
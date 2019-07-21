import _pickle as pickle
import os
import tarfile
from zipfile import *
import glob
import urllib.request as url
import numpy as np
from PIL import Image

def download_test_data():
    
    if not os.path.exists('./k-data/kaggle_test_128.zip'):
        os.mkdir('./k-data')
        print('Start downloading data...')
        url.urlretrieve("https://ecbm4040.bitbucket.io/kaggle_test_128.zip","./k-data/kaggle_test_128.zip")
        print('Download complete.')
    else:
        if os.path.exists('./k-data/kaggle_test_128.zip'):
            print('Kaggle package already exists.')


def load_test_data(mode='all'):
    
    if not os.path.exists('./k-data/kaggle_test_128.zip'):
        download_test_data()
    else:
        print('./k-data/kaggle_test_128.zip already exists. Begin extracting...')
        
    root_dir = os.getcwd()
    os.chdir('./k-data')     
    if not os.path.exists('./test_128'):
        os.system("unzip kaggle_test_128.zip")
    else:
        print("exists already!")
    
    
    test_data = []
    
    extension = ".png"
    path_to_image_folder = "./test_128/"
    num_test_samples = 3500 
    test_img_names = []
    
    
    test_img_names = [path_to_image_folder + str(idx) + extension for idx in range(num_test_samples)]

    
    test_imgages = []
    for img_name in test_img_names:
        im = Image.open(img_name)
        im = im.resize((96,96),Image.ANTIALIAS)
        test_imgages.append(np.array(im))
        im.close()
        
    os.chdir(root_dir)
    X_test = np.asarray(test_imgages)
    return X_test

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:18:27 2018

@author: andrii
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from tqdm import tqdm
import time
from skimage.transform import resize as resize_image

class DataLoader():
    def __init__(self,resize_x,resize_y ):
        self.active = True
        self.real_labels=[]

    def ReadData(self, path):
        """
        Reads data from the  folder. If .pkl file is present uses it as a source.
        If not, scans the folders to create data and lables and stores data in new .pkl file.

        Input:
          path of the unpacked data
        Output:
          Data with lables stored in arrays as tuple.
        """
        os.chdir(path)
        folders=os.listdir()
        if 'data.hdf5' in folders:
            print('Loading data from hdf5 file! Might take some time, be patient!')
            file=h5py.File('data.hdf5','r+')
            data=(np.array(list(file['imgs'])),np.array(list(file['lables'])))
            self.real_labels=list(file['real_labels'])
            file.close()

        else:
            print('1. Collecting data.')
            err_logs = []
            img=[]
            lable=[]
            for folder in tqdm(folders):

                os.chdir(os.path.join(path,folder))
                for file in os.listdir():
                    try:
                        dat=(plt.imread(open(file,'rb')))
                        img.append(resize_image(dat, (resize_x, resize_y),
                                           mode='constant',
                                           ))
                        lable.append(folder)
                        if folder not in self.real_labels:
                            self.real_labels.append(folder)
                      
                    except OSError:
                        err_logs.append([folder, file])
            print('\nError logs:')
            for e in range(len(err_logs)):
                print('\tFolder: {} | Some OSError for file: {}'.format(err_logs[e][0],
                                                                      err_logs[e][0]))
                      
            
            print('2. Encoding data to categorical.')
            # Encode Letters into numerical categories.
            le = LabelEncoder()
            le.fit(lable)
            lable = le.transform(lable)
            lable = np.array(lable).reshape(-1, 1)
          
            print('3. Onehot encoding.')
            # Onehot encoding.
            ohe = OneHotEncoder(sparse=False)
            ohe.fit(lable)
            lable = ohe.transform(lable)
          
            # Shaffle data.
            print('4. Shuffle data.')
            img, lable = shuffle(img, lable)
		  
            print('5. Saving data.')
            data=(np.asarray(img), np.asarray(lable))
            os.chdir(path)
            
            file=h5py.File('data.hdf5','w')
            x=file.create_dataset('imgs',data=np.array(img))
            y=file.create_dataset('lables',data=np.array(lable))
            print(self.real_labels)
            rl=file.create_dataset('real_labels',data=np.string_(self.real_labels))
            file.close()
            print('Data set is stored in Data.hdf5 file. ')

        return data    

    def DataSplit(self, data):
        """
        Standard sklearn train\test spliting. with cross-validation set.

        Input:
            Data tuple
        Output:
            Tuple containing train_X,test_X,valid_X,train_y,test_y,valid_y
        """
        train_X,test_X,train_y,test_y=train_test_split(data[0],data[1], random_state=2)
        valid_X,valid_y=train_test_split(data[0],data[1],random_state=2,test_size=0.15)[1],train_test_split(data[0],data[1],random_state=2,test_size=0.15)[3]
        return (train_X,test_X,valid_X,train_y,test_y,valid_y)
  
    def read_split(self, path):
        print('\nData split.')
        data = self.ReadData(path)
        train_X,test_X,valid_X,train_y,test_y,valid_y, = self.DataSplit(data)
        return train_X,test_X,valid_X,train_y,test_y,valid_y,self.real_labels

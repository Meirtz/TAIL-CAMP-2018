import os
import numpy as np
import struct
import pickle
from PIL import Image
import glob

from keras.callbacks import LambdaCallback
import keras
import random


class read_file_callback(keras.callbacks.Callback):
    def on_train_begin(self,log=None):
        self.data_dir=glob.glob('input/img*.npy')
        self.label_dir=glob.glob('input/label*.npy')
        self.length=len(self.lst_dir)
        self.count=0
        self.data=np.load(self.data_dir[0])
        self.label=np.load(self.label_dir[0])

    def on_epoch_begin(self,epoch,logs=None):
        self.count+=1

    def on_epoch_end(self,epoch,logs=None):
        mod=self.count % self.length
        self.data=np.load(self.data_dir[mod])
        self.label=np.load(self.label_dir[mod])
        index=np.random.permutation(range(len(self.data)))
        self.data=self.data[index]
        self.label=self.label[index]

    def flow_data(self,batch_size):
        i=-1
        while True:
            if i==len(self.data)-batch_size:
                i=-1
            i+=1
            yield self.data[i:i+batch_size],self.label[i:i+batch_size]

rfc=read_file_callback()
model.fit([rfc.data],[rfc.label],batch_size=128,epochs=epoch,callbacks=[rfc],validation_split=0.01)
model.fit_generator(generator=rfc.flow_data(128), steps_per_epoch = int(len(features_train)/batch_size),epochs=500,verbose=1)

if __name__=='__main__':
    pass

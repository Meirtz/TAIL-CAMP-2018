import numpy as np
import cv2
import pandas as pd
import Read_Data
from keras.preprocessing import image

img_raw,label_raw=Read_Data.load_data()

def clean_data(img_raw,label_raw,threshold=25):
    bool_arr=np.array(map(lambda x:True if (x.shape[0]>threshold or x.shape[1]>threshold) else False,img_raw))
    return img_raw[bool_arr],label_raw[bool_arr]

def scale_data(img,full_shape=(128,128)):
    def pad_data(img):
        (fh, fw) = full_shape
        (h, w) = img.shape
        result = np.full(full_shape,255)
        result =result.astype(img.dtype)
        if h>w:
            img=cv2.resize(img,(int(w*fh/float(h)),fh))
            (h, w) = img.shape
            y=0 if (fw-w)==0 else np.random.randint(0,fw-w)
            result[:,y:img.shape[1]+y]=img
        else:
            img=cv2.resize(img,(fw,int(h*fw/float(w))))
            (h, w) = img.shape
            x=0 if (fh-h)==0 else np.random.randint(0,fh-h)
            result[x:img.shape[0]+x,:]=img
        return result.reshape(fh,fw,1)
    return np.array(map(pad_data,img))

class augment_class(object):
    def __init__(self):
        pass

    @staticmethod
    def rotate(x, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
        rotate_limit=(-30, 30)
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    @staticmethod
    def shear(x, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
        intensity=0.5
        sh = np.random.uniform(-intensity, intensity)
        shear=sh
        shear_matrix = np.array([[1, -np.sin(shear), 0],[0, np.cos(shear), 0],[0, 0, 1]])
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
        x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

def augment_data(img):
    img_rotate=np.array(map(augment_class.rotate,img))
    img_shear=np.array(map(augment_class.shear,img))
    return img,img_rotate,img_shear



if __name__=='__main__':
    img=scale_data=scale_data(img_raw)
    augment_data(img)

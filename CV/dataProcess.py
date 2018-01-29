import os
import numpy as np
import struct
import PIL.Image
 
train_data_dir = "HWDB1.1trn_gnt"
test_data_dir = "HWDB1.1tst_gnt"
 
    
import numpy as np
import cv2
import pandas as pd

#img_raw,label_raw=Read_Data.load_data()

def clean_data(img_raw, label_raw, threshold=25):
    bool_arr=np.array(map(lambda x:True if (x.shape[0]>threshold or x.shape[1]>threshold) else False,img_raw))
    return img_raw[bool_arr],label_raw[bool_arr]

def scale_data(img,full_shape=(128,128)):
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
    return result


def read_from_gnt_dir(gnt_dir=train_data_dir):
	def one_file(f):
		header_size = 10
		while True:
			header = np.fromfile(f, dtype='uint8', count=header_size)
			if not header.size: break
			sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
			tagcode = header[5] + (header[4]<<8)
			width = header[6] + (header[7]<<8)
			height = header[8] + (header[9]<<8)
			if header_size + width*height != sample_size:
				break
			image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
			yield image, tagcode
 
	for file_name in os.listdir(gnt_dir):
		if file_name.endswith('.gnt'):
			file_path = os.path.join(gnt_dir, file_name)
			with open(file_path, 'rb') as f:
				for image, tagcode in one_file(f):
					yield image, tagcode
 

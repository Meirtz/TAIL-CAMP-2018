import os
import numpy as np
import struct
import pickle
from PIL import Image

train_data_dir = 'HWDB1.1trn_gnt_small'
test_data_dir = 'HWDB1.1tst_gnt'

def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
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

def load_data():
    img=np.load('img_raw.npy')
    label=np.load('label_raw.npy')
    return img,label

if __name__=='__main__':
    char_set = set()
    img_arr=[]
    label_arr=[]
    for img, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        char_set.add(tagcode_unicode)
        img_arr.append(img)
        label_arr.append(tagcode_unicode)
    char_list = list(char_set)
    char_dict = dict(zip(sorted(char_list), range(len(char_list))))

    with open('char_dict', 'wb') as f:
        pickle.dump(char_dict, f)

    np.save('img_raw.npy',img_arr)
    np.save('label_raw.npy',label_arr)

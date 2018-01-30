from keras.layers import Conv2D,AveragePooling2D,MaxPooling2D,Input,Dense,Dropout,Activation,Lambda
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam,RMSprop,SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import concatenate
from keras.layers.pooling import GlobalAveragePooling2D

import numpy as np
from functools import reduce

class InceptionV3(object):
    def __init__(self):
        pass

    @staticmethod
    def compose(*functions):
        '''
        compose(f,g,h)
        return: h(g(f))
        '''
        composed_function = reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), functions)
        return composed_function

    @staticmethod
    def conv_batchnorm_leaky(filters,kernel_size,strides,padding):
        conv=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,use_bias=False)
        batchnorm=BatchNormalization()
        leaky=LeakyReLU(alpha=0.1)
        return InceptionV3.compose(conv,batchnorm,leaky)

    @staticmethod
    def inception_base_stem():
        return InceptionV3.compose(
            #input:129*129*1
            InceptionV3.conv_batchnorm_leaky(32,(3,3),(2,2),'valid'),
            #output:64*64*32
            InceptionV3.conv_batchnorm_leaky(32,(3,3),(1,1),'same'),
            #output:64*64*32
            InceptionV3.conv_batchnorm_leaky(64,(3,3),(1,1),'valid'),
            #output:62*62*64
            InceptionV3.conv_batchnorm_leaky(64,(3,3),(1,1),'same'),
            #output:62*62*64
            InceptionV3.conv_batchnorm_leaky(80,(3,3),(2,2),'valid')
            #output:31*31*80
            )

    @staticmethod
    def inception_block_a(x):
        #input:31*31*X
        branch1=InceptionV3.conv_batchnorm_leaky(64,(1,1),(1,1),'same')(x)

        branch2=InceptionV3.conv_batchnorm_leaky(32,(1,1),(1,1),'same')(x)
        branch2=InceptionV3.conv_batchnorm_leaky(64,(3,3),(1,1),'same')(branch2)

        branch3=InceptionV3.conv_batchnorm_leaky(64,(1,1),(1,1),'same')(x)
        branch3=InceptionV3.conv_batchnorm_leaky(96,(3,3),(1,1),'same')(branch3)
        branch3=InceptionV3.conv_batchnorm_leaky(96,(3,3),(1,1),'same')(branch3)

        branch4=AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        branch4=InceptionV3.conv_batchnorm_leaky(32,(1,1),(1,1),'same')(branch4)

        result=concatenate([branch1,branch2,branch3,branch4],axis=-1)
        #output:31*31*256
        return result

    @staticmethod
    def subsample_a(x):
        branch1=InceptionV3.conv_batchnorm_leaky(256,(1,1),(1,1),'same')(x)
        branch1=InceptionV3.conv_batchnorm_leaky(416,(3,3),(2,2),'valid')(branch1)

        branch2=InceptionV3.conv_batchnorm_leaky(64,(1,1),(1,1),'same')(x)
        branch2=InceptionV3.conv_batchnorm_leaky(96,(3,3),(1,1),'same')(branch2)
        branch2=InceptionV3.conv_batchnorm_leaky(96,(3,3),(2,2),'valid')(branch2)

        branch3=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(x)
        #output:15*15*768
        result=concatenate([branch1,branch2,branch3],axis=-1)

        return result


    @staticmethod
    def inception_block_b(x):
        branch1=InceptionV3.conv_batchnorm_leaky(192,(1,1),(1,1),'same')(x)

        branch2=InceptionV3.conv_batchnorm_leaky(32,(1,1),(1,1),'same')(x)
        branch2=InceptionV3.conv_batchnorm_leaky(96,(1,5),(1,1),'same')(branch2)
        branch2=InceptionV3.conv_batchnorm_leaky(192,(5,1),(1,1),'same')(branch2)

        branch3=InceptionV3.conv_batchnorm_leaky(64,(1,1),(1,1),'same')(x)
        branch3=InceptionV3.conv_batchnorm_leaky(96,(5,1),(1,1),'same')(branch3)
        branch3=InceptionV3.conv_batchnorm_leaky(96,(1,5),(1,1),'same')(branch3)
        branch3=InceptionV3.conv_batchnorm_leaky(96,(5,1),(1,1),'same')(branch3)
        branch3=InceptionV3.conv_batchnorm_leaky(192,(1,5),(1,1),'same')(branch3)

        branch4=AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        branch4=InceptionV3.conv_batchnorm_leaky(192,(1,1),(1,1),'same')(branch4)
        #output:15*15*768
        result=concatenate([branch1,branch2,branch3,branch4],axis=-1)

        return result

    @staticmethod
    def subsample_b(x):
        branch1=InceptionV3.conv_batchnorm_leaky(192,(1,1),(1,1),'same')(x)
        branch1=InceptionV3.conv_batchnorm_leaky(320,(3,3),(2,2),'valid')(branch1)

        branch2=InceptionV3.conv_batchnorm_leaky(192,(1,1),(1,1),'same')(x)
        branch2=InceptionV3.conv_batchnorm_leaky(192,(1,5),(1,1),'same')(branch2)
        branch2=InceptionV3.conv_batchnorm_leaky(192,(5,1),(1,1),'same')(branch2)
        branch2=InceptionV3.conv_batchnorm_leaky(192,(3,3),(2,2),'valid')(branch2)

        branch3=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(x)
        result=concatenate([branch1,branch2,branch3],axis=-1)
        #output:7*7*1280

        return result

    @staticmethod
    def inception_block_c(x):
        branch1=InceptionV3.conv_batchnorm_leaky(320,(1,1),(1,1),'same')(x)

        branch2=InceptionV3.conv_batchnorm_leaky(384,(1,1),(1,1),'same')(x)
        branch21=InceptionV3.conv_batchnorm_leaky(384,(1,3),(1,1),'same')(branch2)
        branch22=InceptionV3.conv_batchnorm_leaky(384,(3,1),(1,1),'same')(branch2)
        branch2=concatenate([branch21,branch22],axis=-1)

        branch3=InceptionV3.conv_batchnorm_leaky(256,(1,1),(1,1),'same')(x)
        branch3=InceptionV3.conv_batchnorm_leaky(384,(3,3),(1,1),'same')(branch3)
        branch31=InceptionV3.conv_batchnorm_leaky(384,(3,1),(1,1),'same')(branch3)
        branch32=InceptionV3.conv_batchnorm_leaky(384,(1,3),(1,1),'same')(branch3)
        branch3=concatenate([branch31,branch32],axis=-1)


        branch4=AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        branch4=InceptionV3.conv_batchnorm_leaky(192,(1,1),(1,1),'same')(branch4)
        #output:7*7*2048
        result=concatenate([branch1,branch2,branch3,branch4],axis=-1)
        return result

    @staticmethod
    def softmax(x):
        x=InceptionV3.conv_batchnorm_leaky(1024,(1,1),(1,1),'same')(x)
        x=GlobalAveragePooling2D()(x)
        x=Dense(10,activation='softmax')(Dropout(0.9)(x))
        return x

    def build_model(self):
        img_raw=Input((129,129,1))
        img=InceptionV3.inception_base_stem()(img_raw)
        for i in range(3):
            img=InceptionV3.inception_block_a(img)
        img=InceptionV3.subsample_a(img)
        for i in range(3):
            img=InceptionV3.inception_block_b(img)
        img=InceptionV3.subsample_b(img)
        for i in range(3):
            img=InceptionV3.inception_block_c(img)
        output=InceptionV3.softmax(img)
        self.model=Model(inputs=[img_raw],outputs=[output])
        self.model.compile(SGD(lr=1e-4, momentum=0.9, nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])


    def train_model(self,data,epoch):
        img,label=data
        model_checkpoint=ModelCheckpoint('models/model.hdf5',monitor='loss',verbose=1,save_best_only=True)
        self.model.fit([img],[label],batch_size=128,epochs=epoch,callbacks=[model_checkpoint],validation_split=0.1)

if __name__=='__main__':
    instance=InceptionV3()
    instance.build_model()
    img=np.load('mnist.npy')
    label=np.load('mnist_label.npy')
    instance.train_model((img,label),100)

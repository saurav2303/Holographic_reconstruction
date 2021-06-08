"""""MODEL BUILDING""""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, ReLU, Lambda,MaxPool2D
from tensorflow.python.keras.models import Model

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

###################### subpixel convolution #####################################

def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(64 * (factor ** 2), 3, padding='same', **kwargs)(x)
        #x = Conv2D(64 , 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

 def upsample(x):
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
        x = upsample_1(x, 2, name='conv2d_3_scale_2')
        return x 

 ############## Residual Block ##############3#################################

def res_block(x_in, filters,pool, skip,num):
     
    
          if (pool) :
            x = MaxPool2D (pool_size=(2, 2),strides=(2, 2), padding='valid',name='Res_maxpool_'+str(num))(x_in)
            x = Conv2D(filters, 3, padding='same',name='Res_conv2D_first'+str(num) )(x)
            x = BatchNormalization(axis=-1, momentum=0.99 ,epsilon=0.001,name='Res_BN_first'+str(num))(x)
            x = ReLU (name='Res_ReLU_first'+str(num))(x)
            x = Conv2D(filters, 3, padding='same',name='Res_conv2D_second'+str(num) )(x)
            x = BatchNormalization(axis=-1, momentum=0.99 ,epsilon=0.001,name='Res_BN_second'+str(num))(x)
            x = ReLU (name='Res_ReLU_second'+str(num))(x)
            x_in  = MaxPool2D (pool_size=(2, 2),strides=(2, 2), padding='valid',name='Res_poolSKIP_'+str(num))(x_in)
            shortcut = Conv2D(skip, 1, padding='same',name='Res_conv2DSKIP_'+str(num) )
            x_in = shortcut(x_in)
            x = Add()([x_in, x])
          else :
            x = Conv2D(filters, 3, padding='same',name='Res_conv2D_first'+str(num) )(x_in)
            x = BatchNormalization(axis=-1, momentum=0.99 ,epsilon=0.001,name='Res_BN_first'+str(num))(x)
            x = ReLU (name='Res_ReLU_first'+str(num))(x)
            x = Conv2D(filters, 3, padding='same',name='Res_conv2D_second'+str(num) )(x)
            x = BatchNormalization(axis=-1, momentum=0.99 ,epsilon=0.001,name='Res_BN_second'+str(num))(x)
            x = ReLU (name='Res_ReLU_second'+str(num))(x)
            shortcut = Conv2D(skip, 1, padding='same',name='Res_conv2DSKIP_'+str(num) )
            x_in = shortcut(x_in)
            x = Add()([x_in, x])
    
          return x
    
    ############### HR-net #########################################################
    
    def HRNet( num_filters=32):
    """Creates an HRNet model."""
    x_in = Input(shape=(500, 500,3))
    
    x = Conv2D(num_filters, 3, padding='same',name='First_conv2D')(x_in)
    x = BatchNormalization(axis=-1, momentum=0.99 ,epsilon=0.001,name='First_BN')(x)
    x = ReLU (name='First_RELU')(x)
    
    
    x = res_block(x,filters=64,pool=1,skip=64,num=1)
    x  = res_block(x,filters=64,pool=0,skip=64,num=2)
    x = res_block(x,filters=128,pool=1,skip=128,num=3)
    x = res_block(x,filters=128,pool=0,skip=128,num=4)
    #x = res_block(x,filters=256,pool=1,skip=256,num=5)
    #x = res_block(x,filters=256,pool=0,skip=256,num=6)

    

    x = upsample_1(x,factor=4)
    x =x = Conv2D(3, 3, padding='same',name='Last_conv2D')(x)
    

    
    model=Model(x_in, x, name="HRNet")
    return model
  
  #################### loading training data ########################################
  
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
    
def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

def list_array(list):
    Array = np.array(list)
    return Array
ef load_data_from_dirs(path):
    files = []
    file_names = []
    count = 0
    z = os.listdir(path)
    z.sort()
    
    for f in z : 
        
            image = cv2.imread(os.path.join(path,f))
            
            files.append(image)
            file_names.append(os.path.join(path,f))
            count = count + 1
    return files     


    
def load_training_data(path_in, path_out, number_of_images = 800, train_ratio = 0.8,val_ratio =0.1):

    number_of_train_images = int(number_of_images * train_ratio)
    number_of_val_images = int(number_of_images * val_ratio)
    
    files_in = load_data_from_dirs(path_in)
    files_out = load_data_from_dirs(path_out)

    x_train = files_in[:number_of_train_images]
    x_val = files_in[number_of_train_images:(number_of_train_images+number_of_val_images)]

    y_train = files_out[:number_of_train_images]
    y_val = files_out[number_of_train_images:(number_of_train_images+number_of_val_images)]

    X_train =list_array (x_train)
    X_val = list_array (x_val)
    Y_train = list_array (y_train)
    Y_val = list_array (y_val)

    return normalize(X_train) , normalize(Y_train) ,normalize (X_val) ,normalize(Y_val)
    
 path2='/content/drive/MyDrive/Colab Notebooks/HRNET/directoryTooutput'
 path1 ='/content/drive/MyDrive/Colab Notebooks/HRNET/dirctoryToinput'

 x_train,y_train,x_val,y_val = load_training_data(path1,path2)

################### training , validation , saving the model ##################################


import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# Create directory for saving model weights
weights_dir = '/content/drive/MyDrive/Colab Notebooks/HRNET'
os.makedirs(weights_dir, exist_ok=True)


model_HRNet = HRNet(num_filters=32)

# Adam optimizer with a scheduler that halfs learning rate after 200,000 steps
optim_HRNet = Adam(learning_rate=0.001)

# Compile and train model for 300,000 steps with L1 pixel loss
model_HRNet.compile(optimizer=optim_HRNet, loss='mean_squared_error')
model_HRNet.fit(x_train,y_train, epochs=1, batch_size =32 ,validation_data=(x_val, y_val))

## train_ds is our dataset

# Save model weights
model_HRNet.save_weights(os.path.join(weights_dir, 'weights-HRNet-16-x4.h5'))
    
  
 

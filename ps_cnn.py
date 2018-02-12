import numpy as np
import tensorflow as tf
import pdb
import scipy.io
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D,Flatten
import glob2
from sklearn.model_selection import train_test_split
import csv
import math

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import LearningRateScheduler , TensorBoard
from keras import initializers


data_path= '/home/rnd7528/git/plant_seed_classification/'
#label_file = './preprocessing/abide_labels.csv'
#all_filenames = glob2.glob(data_path + '/**/*.mat')
#no_files = len(all_filenames)

def get_data(filenames):

    data = []
    for name in filenames:
    	
        matFile = scipy.io.loadmat(name)
        X = matFile['connectivity']
        data.append(X)
    
    return data

	
def get_labels(label_file):
    
    with open(label_file, 'r+') as f:
        reader = csv.reader(f, delimiter = ',')
        labels = [rows[1] for rows in reader]
		
    return labels
	
# def cnn_block(input, name, filter = 64, kernel = [3,3], stride = (1,1), activation='relu', pooling='max' pool_size= (2,2), pool_stride = (2,2)):

    # x = Conv2D(filter, kernel, activation='relu', padding='same', name=name+'_conv1')(input)
    # x = Conv2D(filter, kernel, activation='relu', name=name+'_conv2')(x)
    # if pooling = 'avg':
	    # x = .AveragePooling2D(pool_size, strides=pool_stride, name=name+'_avgpool')(x)
    # else:
        # x = MaxPooling2D(pool_size, strides=pool_stride, name=name+'_maxpool')(x)
    # return x
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
	
def get_model(batch_size):

    #pdb.set_trace()
    model = Sequential()
    model.add(Conv2D(256,  kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'same', input_shape=(3,224,224)))
    model.add(Conv2D(512,kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, kernel_size = (3,3), activation='relu',padding = 'same'))
    model.add(Conv2D(2048,kernel_size = (3,3), activation='relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512,kernel_initializer='he_normal', kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'same'))
    model.add(Conv2D(256,kernel_initializer='he_normal',kernel_size=(3, 3), strides=(1, 1), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3,3), activation='relu',padding = 'same'))
    model.add(Conv2D(64, (3,3), activation='relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer='he_normal',activation='relu'))
    model.add(Dense(12, activation='softmax'))

    return model	
	

def get_gen(path,target_size = (150, 150),batch_size=50,isTrain=1):
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
		
    if isTrain:
        generator = train_datagen.flow_from_directory(
			path+'train',  # this is the target directory
			target_size=target_size,  # all images will be resized to 150x150
			batch_size=batch_size,
			class_mode='categorical')
    else:
	    generator = test_datagen.flow_from_directory(
			path+'test',
			target_size=target_size,
			batch_size=batch_size,
			class_mode='categorical')
    return generator			

def main():

    #all_filenames = glob2.glob(data_path + '/**/*.mat')
    #labels= get_labels(label_file)
    #print('Labels Loaded')	
    batch_size = 5
    # train_data = get_data(all_filenames)
    # train_data = np.reshape(train_data, (len(train_data),1,111,111))
    # train_labels = np.array(labels)
	
    train_gen = get_gen(data_path,(224,224),batch_size,1)
    test_gen = get_gen(data_path,(224,224),batch_size,0)
    print('Data Loaded')
    model = get_model(batch_size)

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

    lrate = LearningRateScheduler(step_decay)
    tboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [lrate,tboard]

    print('Model Fitting started.')
    model.fit_generator(train_gen,
         steps_per_epoch=5, #2000 // batch_size,
		 epochs=1,
		 callbacks=callbacks_list)
    model.save_weights('plant_seed_model.h5')
    # pred = model.predict_generator(test_gen,steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    # test_names = glob2.glob(data_path+'test/*.png')
    # with open('./submit.csv', 'wb') as f:
        # writer = csv.writer(f, delimiter=',')
        # writer.writerow('file,species')
        # for i in len(test_names):
            # name = test_name[i].split('/')[-1]
            # writer.writerow(name,pred[i])
    # print('Predictions Printed')
            

	# save model 
 #get model predictions
 
main()
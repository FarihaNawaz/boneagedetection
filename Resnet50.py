# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:59:24 2019

@author: 15301121
"""

import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as sk_mae
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dont allocate entire vram initially
set_session(tf.Session(config=config))
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50,preprocess_input 
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from keras.models import Model
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("Reading data")
img_dir = "boneage-training-dataset/"
csv_path = "boneage-training-dataset.csv"
age_df = pd.read_csv(csv_path)
age_df['path'] = age_df['id'].map(lambda x: img_dir+"{}.png".format(x))
age_df['exists'] = age_df['path'].map(os.path.exists)
age_df['gender'] = age_df['male'].map(lambda x: "male" if x else "female")
mu = age_df['boneage'].mean()
sigma = age_df['boneage'].std()
age_df['zscore'] = age_df['boneage'].map(lambda x: (x-mu)/sigma)
age_df.dropna(inplace=True)
print("{} images found out of total {} images".format(age_df['exists'].sum(),age_df.shape[0]))
print("Reading complete !!!\n")
print("Preparing training, testing and validation datasets ...")
age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
raw_train_df, test_df = train_test_split(age_df, 
                                   test_size = 0.2, 
                                   random_state = 2019,
                                   stratify = age_df['boneage_category'])
raw_train_df, valid_df = train_test_split(raw_train_df, 
                                   test_size = 0.1,
                                   random_state = 2019,
                                   stratify = raw_train_df['boneage_category'])
train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace = True)).reset_index(drop=True)
train_size = train_df.shape[0]
valid_size = valid_df.shape[0]
test_size = test_df.shape[0]
print("# Total training images:   {}".format(train_size))
print("# Total validation images: {}".format(valid_size))
print("# Total testing images:    {}".format(test_size))
IMG_SIZE = (224, 224) 
def datagen(in_df, path_col, y_col, gender_col, batch_size, **dflow_args):
    img_data_gen = ImageDataGenerator(samplewise_center=False, 
                            samplewise_std_normalization=False, 
                            horizontal_flip = True, 
                            vertical_flip = False, 
                            height_shift_range = 0.15, 
                            width_shift_range = 0.15, 
                            rotation_range = 5, 
                            shear_range = 0.01,
                            fill_mode = 'reflect',
                            zoom_range=0.25,
                           preprocessing_function = preprocess_input)
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse', batch_size = batch_size, shuffle=True, **dflow_args)
    df_gen.filenames = [x.split("/")[1] for x in in_df[path_col].values]
    df_gen.classes = np.column_stack([in_df[y_col].values,in_df[gender_col].values.astype(float)])
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    for batch in df_gen:
      yield [batch[0],batch[1][:,1]],batch[1][:,0]
train_gen = datagen(train_df, 
                     path_col = 'path',
                    y_col = 'zscore',
                    gender_col = 'male',
                    batch_size = 10,
                    target_size = IMG_SIZE,
                    color_mode = 'rgb',
                    seed=8309)
print("Preparing validation image")
valid_X, valid_Y = next(datagen(valid_df, 
                                 path_col = 'path',
                                y_col = 'zscore',
                                gender_col = 'male',
                                batch_size = valid_size,
                                target_size = IMG_SIZE,
                                 color_mode = 'rgb',
                                 seed=8309)) # one big batch
IMG_SHAPE = valid_X[0][0,:,:,:].shape
print("Image shape: "+str(IMG_SHAPE))
print("All data prepared !!!\n")
print("Compiling deep model ")
img = Input(shape = IMG_SHAPE)
gender = Input(shape=(1,))
cnn_vec = ResNet50(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')(img)
cnn_vec = GlobalAveragePooling2D()(cnn_vec)
cnn_vec = Dropout(0.2)(cnn_vec)
gender_vec = Dense(32,activation='relu')(gender)
features = Concatenate(axis=-1)([cnn_vec,gender_vec])
dense_layer = Dense(1024, activation = 'relu')(features)
dense_layer = Dropout(0.2)(dense_layer)
dense_layer = Dense(1024,activation='relu')(dense_layer)
dense_layer = Dropout(0.2)(dense_layer)
output_layer = Dense(1, activation = 'linear')(dense_layer) # linear is what 16bit did
bone_age_model = Model(inputs=[img,gender],outputs=output_layer)
def mae_months(in_gt, in_pred):
    return mean_absolute_error(mu+sigma*in_gt, mu+sigma*in_pred)
bone_age_model.compile(optimizer = 'adam', loss = 'mse', metrics = [mae_months])
bone_age_model.summary()
print("Model compiled\n")
print("Training deep model ...")
weight_path="resnet.best.hdf5" # saved_model_name
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
if not os.path.exists(weight_path):
    bone_age_model.fit_generator( train_gen,
                                  steps_per_epoch = train_size/100,
                                  validation_data = (valid_X,valid_Y),
                                  epochs = 10, 
                                  callbacks = callbacks_list,
                                  verbose=1)
bone_age_model.load_weights(weight_path)
print("Training complete !!!\n")
print("Evaluating model on test data\n")
print("Preparing testing dataset")
test_X, test_Y = next(datagen(test_df, 
                             path_col = 'path',
                            y_col = 'zscore',
                            gender_col = 'male',
                            batch_size = test_size,
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                             seed=8309))
print("Data prepared")
pred_Y = mu+sigma*bone_age_model.predict(x=test_X,batch_size=25,verbose=1)
test_Y_months = mu+sigma*test_Y
print("Mean absolute error on test data: "+str(sk_mae(test_Y_months,pred_Y)))
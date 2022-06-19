'''
===============================================================================================
training.py
Dependencies: tensorflow, h5py; module buildup.py.
Requirements for main program: the 'NeatSynthText.h5' file produced by 'buildup.py', being in
    the current working directory.

Nadav Kahlon, January 2022.
===============================================================================================

This program is responsible for creating and training CNN models to classify fonts of
characters appearing in given images.

The input to those models is a 64-by-64 pixels concentrated images of individual characters,
as produced by the 'concentrateImg' function (defined in buildup.py)
The output is a 'total_fonts'-vector of values between 0 and 1. The values correspond probability
that each one of the 'total_fonts' fonts (defined in the 'fonti_map' in 'buildup.py') appears in
the image.

Fitting and validation is done using the dataset in the HDF5 file 'NeatSynthText.h5', as
created by the 'buildup' program.

Different models are fit to classify fonts of different charcters. Therefore - Models
will be created only to the characters appearing in the train and validation sets. All
models are stored inside a local directory 'models' in TensorFlow SavedModel format,
named by the ASCII value of the corresponding character.

Also - a global model trained on the entire dataset is stored in the 'models' directory,
as 'global'. This can be used to classify characters that did not appear in the original
train set.
Moreover - all the "inidividual character" models are actually fine-tuned versions of
the global one.

An "individual character" model will not be saved in one of the following cases:
    > In case there are less than 30 training example to fit it on.
    > In case the validation loss of the created model is less than the validation
        loss of the global model, for the same validation set.
The heuristics behind these decisions are explained in the attached report.

An additional HDF5 file named 'weights.h5' is created inside the 'models' directory.
It contains 1-value-datasets named after the ASCII values of the characters having a 
model in the directory (+ a 'global' entry for the global model).
Weights are calculated using the following formula:
    (Accuracy / (2-Accuracy))
The formula, and the heuristics behind it, are explained in detail in the attached report.

'''

from buildup import total_fonts
from keras.regularizers import l2
import tensorflow as tf
from tensorflow.image import random_contrast, random_brightness, random_saturation, random_hue
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session as keras_clear_session
from tensorflow.keras.callbacks import ReduceLROnPlateau
import gc
import os
import h5py


'''
class RandomColorDistortion : a custom CNN layer for producing random-colored variants
    of the training examples in the dataset, before each epoch of the fitting process.
'''
class RandomColorDistortion(tf.keras.layers.Layer):
    '''
    Constructor : creates a random color distortion CNN layer, producing random contrast,
        brightness and saturation in the given ranges,c and random hue between -'hue_delta'
        and 'hue_delta'.
    '''
    def __init__(self, contrast_range=(0,2), brighness_range=(0.5,1.5), saturation_range=(0.5,1.5), hue_delta=0.5, **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.contrast_range = contrast_range
        self.brighness_range = brighness_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
    
    '''
    call : distorts the color of the given images (if 'training' is True).
    '''
    def call(self, images, training=None):
        if training:
          images = random_contrast(images, self.contrast_range[0], self.contrast_range[1])
          images = random_brightness(images, self.brighness_range[0], self.brighness_range[1])
          images = random_saturation(images, self.saturation_range[0], self.saturation_range[1])
          images = random_hue(images, self.hue_delta)
          images = tf.clip_by_value(images, 0, 1)
        return images


'''
createModel : creates and compiles a new CNN model suited for classifying fonts in
    concentrated images of charcaters.
'''
def createModel():
    # the architecture I developed:
    model = Sequential([
        InputLayer(input_shape=(64,64,3)),
        RandomColorDistortion(name='randomcolordistortion'),
    
        Conv2D(filters=64,kernel_size=(9,9), kernel_regularizer=l2(0.0005), activation=tf.keras.layers.LeakyReLU(alpha=0.5), padding='same'),
        MaxPooling2D(pool_size=(2,2)),
    
        Conv2D(filters=32,kernel_size=(6,6), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
    
        Conv2D(filters=64,kernel_size=(3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
    
        Flatten(),
        Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        Dropout(0.25),
        Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
        Dropout(0.25),
        Dense(total_fonts, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

'''
Main program : create, fit, and store the models descibed at the top of the file.
Requirements: the 'NeatSynthText.h5' file produced by 'buildup.py', being in the current
    working directory.
Produces: 'models' directory, and all its contents described at the top of this file. 
'''
def training_main():
    # create the 'models' directory (if it does not exist)
    if not os.path.exists('models'):
        os.mkdir('models')
    
    # create the HDF5 file 'weights.h5' mentioned above
    with h5py.File('models/weights.h5', 'w') as weights_file:
        # load the dataset
        with h5py.File('NeatSynthText.h5', 'r') as db:
            
            # first - we fit a global model
            # import the whole dataset
            X_train = db['train']['data']['images'][()]
            y_train = db['train']['data']['fontis'][()]
            X_val = db['val']['data']['images'][()]
            y_val = db['val']['data']['fontis'][()]
            # create, compile, fit, and save the model
            print('Fitting global model')
            global_model = createModel()
            lr_decay = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, mode="auto",
                                         verbose=1, min_delta=0.0001,cooldown=0, min_lr=0)
            global_model.fit(x=X_train, y=y_train, epochs=130, shuffle=True,
                             validation_data=(X_val,y_val), callbacks=[lr_decay])
            global_model.save('models/global')
            # calculate and store its weight
            print('Evaluating global model')
            loss, acc = global_model.evaluate(X_val, y_val)
            weights_file.create_dataset('global', data=acc/(2-acc))
            # cleanup
            keras_clear_session()
            gc.collect()
            
            # next - create a model for every charcater in the training and validation sets
            # characters we are going to examine
            chars = [char for char in db['train']['indexing']['by_char'].keys()
                     if char in db['val']['indexing']['by_char'].keys()]
            
            for char_i, char in enumerate(chars):
                print('Trying to fit model for character:', char, '|', char_i+1, '/', len(chars))
                
                # load training and validation sets
                train_indices = db['train']['indexing']['by_char'][char][()]
                X_train = db['train']['data']['images'][train_indices]
                y_train = db['train']['data']['fontis'][train_indices]
                val_indices = db['val']['indexing']['by_char'][char][()]
                X_val = db['val']['data']['images'][val_indices]
                y_val = db['val']['data']['fontis'][val_indices]
                
                if len(y_train) >= 30: # we fit a model only when there are at least 30 examples
                    
                    curr_model = load_model('models/global') # new models are fine-tuned versions of the global one            
                    curr_model.compile(optimizer=tf.keras.optimizers.Adam(0.00025),
                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                       metrics=['accuracy'])
                    lr_decay = ReduceLROnPlateau(monitor="val_loss", factor=0.5**0.5, patience=10, mode="auto",
                                                 verbose=1, min_delta=0.0001,cooldown=0, min_lr=0)
                    curr_model.fit(x=X_train, y=y_train, epochs=65, shuffle=True,
                                   validation_data=(X_val,y_val), callbacks=[lr_decay])
                    
                    # evaluate the resulting model and the global model on validation set
                    print('Evaluating model for character:', char, '|', char_i+1, '/', len(chars))
                    local_loss, local_acc = curr_model.evaluate(X_val, y_val)
                    global_loss, global_acc = global_model.evaluate(X_val, y_val)
                    
                    if local_loss < global_loss: # we save a model only if it improved over the global one
                        curr_model.save('models/' + char)
                        weights_file.create_dataset(char, data=local_acc/(2-local_acc))
                        
                    # cleanup
                    keras_clear_session()
                    del(curr_model)
                    gc.collect()
                
                else:
                    print('Skipped (not enough training examples).')

if __name__ == '__main__':
    training_main()
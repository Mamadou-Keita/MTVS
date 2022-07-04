import tensorflow as tf
from keras.applications.vgg16 import VGG16,  preprocess_input
import numpy as np
import pandas as pd
from keras import backend as K 
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D, Dense, Activation, BatchNormalization, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
import warnings
import os
import argparse
from keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#Train using GPU
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def VisualQuality(x):

  layer = Dense(512, kernel_initializer='normal')(x)
  layer = BatchNormalization()(layer)
  layer = Activation('relu')(layer)
  layer = Dropout(0.2)(layer)

  layer = Dense(512, kernel_initializer='normal')(x)
  layer = BatchNormalization()(layer)
  layer = Activation('relu')(layer)
  layer = Dropout(0.2)(layer)

  imgOut = Dense(1, name='VisualQuality')(layer)

  return imgOut

def VisualSecurity(x):
  layer = Dense(512, kernel_initializer='normal')(x)
  layer = BatchNormalization()(layer)
  layer = Activation('relu')(layer)
  layer = Dropout(0.2)(layer)

  layer = Dense(512, kernel_initializer='normal')(layer)
  layer = BatchNormalization()(layer)
  layer = Activation('relu')(layer)
  layer = Dropout(0.2)(layer)

  visualOut = Dense(1, name='VisualSecurity')(layer)

  return visualOut

def _train(args):

    train = pd.read_csv(args.input)
    print(train.head())

    trainedModel = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(trainedModel.output)

    for layer in trainedModel.layers[:3]:
        layer.trainable = False
        print(layer)

    for layer in trainedModel.layers[3:]:
        layer.trainable = True

    model = Model(inputs =trainedModel.input, outputs= [VisualQuality(x), VisualSecurity(x)], name='MultiTaskRegression')
    print(model.summary())

    datagen = ImageDataGenerator(
        rescale= 1/255.,
        horizontal_flip= True,
        vertical_flip= True,
        # preprocessing_function = preprocess_input
     )

    train_generator = datagen.flow_from_dataframe(
        dataframe= train, 
        directory= args.data, 
        x_col='images', 
        y_col=['image_score','visual_score'], 
        class_mode="raw", 
        target_size=(224,224), 
        batch_size=64,
        shuffle= True
    )

    model.compile(loss= 'mse', optimizer=tf.optimizers.Adam(0.0001))

    callbak = [tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience= 50,
        mode = 'auto',
        min_delta=1e-3,
        restore_best_weights = True,
        verbose =1
    )]
    history = model.fit(
        train_generator,
        steps_per_epoch= len(train) // (4*16),
        epochs=1500,
        callbacks = callbak
    )

    model.save('VGG16MSEMultiTask2.h5')

    plt.plot(history.history['loss'])
    plt.savefig('loss.png')
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model Training...')
    parser.add_argument("--data", "-d", help="path to the data", type=str, default='./patches')
    parser.add_argument("--input", "-i", help="input csv file", type=str, default='train.csv')
    args = parser.parse_args()

    _train(args)
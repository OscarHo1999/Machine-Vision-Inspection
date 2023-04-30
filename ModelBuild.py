import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import scipy
import statistics
import os


def ModelBuild(directory, modelpath):

    base_model = InceptionV3(input_shape=(256, 256, 3), include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    X = Flatten()(base_model.output)
    X = Dense(units=2, activation='sigmoid')(X)

    # Final Model
    model = Model(base_model.input, X)
    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    # model.summary()

    # preprocess data using data generator
    train_datagen = ImageDataGenerator(featurewise_center=True, rotation_range=0.4, width_shift_range=0.3,
                                       horizontal_flip=True, preprocessing_function=preprocess_input, zoom_range=0.4,
                                       shear_range=0.4)
    train_data = train_datagen.flow_from_directory(directory=directory,
                                                   target_size=(256, 256), batch_size=36)
    # bad=0 good=1
    # print(train_data.class_indices)

    # Model Check Point
    filepath = "model"+"\\"+modelpath+".h5"
    mc = ModelCheckpoint(filepath=filepath, monitor="accuracy", verbose=1, save_best_only=True)
    es = EarlyStopping(monitor="accuracy", min_delta=0.0001, patience=5, verbose=1)
    cb = [mc, es]

    his = model.fit(train_data, steps_per_epoch=5, epochs=2, callbacks=cb)

    h = his.history
    mean_loss = "{:.2f}".format(statistics.mean(h['loss'])*100)
    mean_accuracy = "{:.2f}".format(statistics.mean(h['accuracy'])*100)
    improve_loss = "{:.2f}".format(h['loss'][-1]*100)
    improve_accuracy = "{:.2f}".format(h['accuracy'][-1]*100)

    mean_loss = str(mean_loss)+"%"
    mean_accuracy = str(mean_accuracy)+"%"
    improve_loss = str(improve_loss)+"%"
    improve_accuracy=str(improve_accuracy)+"%"

    plot_path = "plot"+"/"+modelpath+".png"
    plt.plot(h['loss'])
    plt.plot(h['accuracy'], 'go--', c="red")
    plt.title("Loss vs Acc")
    plt.ylabel('Data')
    plt.xlabel('Epochs')
    plt.legend(['loss', 'accuracy'],loc='upper right')
    plt.savefig(plot_path)
    plt.show()

    train_data_mean_loss_path = "train_data_path"+"/"+modelpath+"/"+"loss"
    train_data_mean_accuracy_path = "train_data_path"+"/"+modelpath+"/"+"accuracy"
    train_data_improve_loss_path = "train_data_path"+"/"+modelpath+"/"+"improve loss"
    train_data_improve_accuracy_path = "train_data_path"+"/"+modelpath+"/"+"improve accuracy"

    os.makedirs(train_data_mean_loss_path)
    os.makedirs(train_data_mean_accuracy_path)
    os.makedirs(train_data_improve_loss_path)
    os.makedirs(train_data_improve_accuracy_path)


    f1 = open(train_data_mean_loss_path+"/"+"loss.txt", 'a')
    f1.write(mean_loss)
    f1.close()

    f2 = open(train_data_mean_accuracy_path+"/"+"accuracy.txt",'a')
    f2.write(mean_accuracy)
    f2.close()

    f3 = open(train_data_improve_loss_path+"/"+"loss.txt", 'a')
    f3.write(improve_loss)
    f3.close()

    f4 = open(train_data_improve_accuracy_path+"/"+"accuracy.txt",'a')
    f4.write(improve_accuracy)
    f4.close()








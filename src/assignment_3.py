# operating system
import os 

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

# import argparse
import argparse

# import pandas
import pandas as pd

def mdl(learn_rate, dec_steps, dec_rate):
    # initialising the model
    model = VGG16(include_top = False,
                  pooling = "avg",
                  input_shape = (32, 32, 3))
    # desabling convolutional layers (?)
    for layer in model.layers:
        layer.trainable = False
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = "relu")(flat1)
    output = Dense(10, activation = "softmax")(class1)
    
    #define the new model
    model = Model(inputs = model.inputs,
                  outputs = output)
    
    #compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = float(learn_rate),
        decay_steps = int(dec_steps),
        decay_rate = float(dec_rate))
    sgd = SGD(learning_rate = lr_schedule)
    model.compile(optimizer = sgd,
                  loss = "categorical_crossentropy",
                  metrics = ["accuracy"])
    return model

def load_process_data():
    # load
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # normalize
    X_train_norm = X_train/255
    X_test_norm = X_test/255
    
    # create one-hot encodings
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    # initialize label names for CIFAR-10 dataset
    label_names = ["airplane", 
                   "automobile", 
                   "bird", 
                   "cat", 
                   "deer", 
                   "dog", 
                   "frog", 
                   "horse", 
                   "ship", 
                   "truck"]
    return X_train_norm, X_test_norm, y_train, y_test, label_names

#train the model and return H
def train_model(model, X_train_norm, y_train, X_test_norm, y_test, b_size, eps):
    H = model.fit(X_train_norm, y_train,
                  validation_data = (X_test_norm, y_test),
                  batch_size = int(b_size),
                  epochs = int(eps),
                  verbose = 1)
    return H

# define the plotting function we saw in class
def plot_history(H, eps, vis_name): 
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, int(eps)), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, int(eps)), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, int(eps)), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, int(eps)), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    outpath_vis = os.path.join("out", vis_name)
    fig = plt.savefig(outpath_vis, dpi = 300, bbox_inches = "tight")
    return fig


def classification_rep(model, X_test_norm, y_test, b_size, rep_name):
    #get your predictions
    predictions = model.predict(X_test_norm, int(b_size)) 
    # make the classification report
    report = classification_report(y_test.argmax(axis = 1), predictions.argmax(axis = 1)) 
    outpath_rep = os.path.join("out", rep_name)
    # write the report to a txt file
    with open(outpath_rep,"w") as file:
        file.write(str(report)) 



def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-e", "--eps", required=True, help="the number of epochs")
    ap.add_argument("-vn", "--vis_name", required=True, help="the name of the figure")
    ap.add_argument("-lr", "--learn_rate", required=True, help="the learning rate of the model")
    ap.add_argument("-ds", "--dec_steps", required=True, help="the decay steps of the model")
    ap.add_argument("-dr", "--dec_rate", required=True, help="the decay rate of the model")
    ap.add_argument("-bs", "--b_size", required=True, help="the batch size of the model")
    ap.add_argument("-rp", "--rep_name", required=True, help="the name of the classification report")
    args = vars(ap.parse_args())
    return args

def main():
    args = parse_args()
    model = mdl(args["learn_rate"], args["dec_steps"], args["dec_rate"])
    X_train_norm, X_test_norm, y_train, y_test, label_names = load_process_data()
    H = train_model(model, X_train_norm, y_train, X_test_norm, y_test, args["b_size"], args["eps"])
    fig = plot_history(H, args["eps"], args["vis_name"])
    report_df = classification_rep(model, X_test_norm, y_test, args["b_size"], args["rep_name"])
    
if __name__ == "__main__":
    main()
    


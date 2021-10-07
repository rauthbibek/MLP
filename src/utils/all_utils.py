import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import joblib
import os
import yaml

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content

def create_dirs(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Directory is created at {dir}")

def preapare_data(validation_datasize):

    mnist = tf.keras.datasets.mnist

    (X_train_full,y_train_full),(X_test, y_test)=mnist.load_data()

    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    return (X_train,y_train),(X_test, y_test),(X_valid, y_valid)

def save_model(model, filepath):

    #joblib.dump(model, filepath)
    model.save(filepath)

def save_plot(history, plotpath):
    
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(plotpath)









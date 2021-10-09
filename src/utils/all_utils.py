import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
import yaml
import time

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

def get_unique_file_names(file_name):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{file_name}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename =  get_unique_file_names(model_name)
    model_path = os.path.join(model_dir, unique_filename)
    model.save(model_path)

def save_plot(history, plotname, plot_dir,):
    unique_filename = get_unique_file_names(plotname)
    plot_path = os.path.join(plot_dir, unique_filename)
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(plot_path)









import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import joblib
import os



def data():

    return tf.keras.datasets.mnist

def preapare_data():

    (X_train_full,y_train_full),(X_test, y_test)=data().load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    return (X_train,y_train),(X_test, y_test),(X_valid, y_valid)

def save_model(model, file_name):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, file_name)
    joblib.dump(model, filepath)

def save_plot(history, file_name):
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plotpath = os.path.join(plot_dir, file_name)
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(plotpath)









#from utils.constants import INPUT_SHAPE, METRICS, N_CLASS, EPOCHS, LOSS_FUNCTION, OPTIMIZER, METRICS

import tensorflow as tf


class MLPerceptron:

    def __init__(self, input_shape, epochs, n_class):
        self.input_shape = input_shape
        self.epochs = epochs
        self.n_class = n_class
        
    def classifier(self):
        LAYERS = [
          tf.keras.layers.Flatten(input_shape=self.input_shape, name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(self.n_class, activation="softmax", name="outputLayer")]
        
        return tf.keras.models.Sequential(LAYERS)

    def model_compile(self,model_clf, LOSS_FUNCTION, OPTIMIZER, METRICS):

        model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
        return model_clf

    def model_fit(self, model_clf, X_train, y_train, X_valid, y_valid):

        history = model_clf.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_valid, y_valid))
        return history

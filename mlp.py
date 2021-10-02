from utils.constants import INPUT_SHAPE, METRICS, N_CLASS, EPOCHS, LOSS_FUNCTION, OPTIMIZER
from utils.all_utils import  preapare_data, save_model, save_plot
from utils.models import MLPerceptron

def main():

    (X_train,y_train),(X_test, y_test),(X_valid, y_valid)=preapare_data()
    mlp_obj = MLPerceptron(input_shape=INPUT_SHAPE, epochs=EPOCHS, n_class= N_CLASS)
    model_clf = mlp_obj.classifier()
    print(model_clf.summary())
    _ = mlp_obj.model_compile(model_clf,LOSS_FUNCTION, OPTIMIZER, METRICS)
    history = mlp_obj.model_fit(model_clf, X_train, y_train, X_valid, y_valid)
    save_model(model_clf, "model01.model")
    save_plot(history=history, file_name="Loss_vs_accuracy.png")


if __name__ == "__main__":
    main()






    
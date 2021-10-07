#from utils.constants import INPUT_SHAPE, METRICS, N_CLASS, EPOCHS, LOSS_FUNCTION, OPTIMIZER
from utils.all_utils import  preapare_data, save_model, save_plot, read_config, create_dirs
from utils.models import MLPerceptron
import os
import argparse

def main(config_path):

    config = read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]

    (X_train,y_train),(X_test, y_test),(X_valid, y_valid)=preapare_data(validation_datasize)
    INPUT_SHAPE = config["params"]["INPUT_SHAPE"]
    EPOCHS = config["params"]["EPOCHS"]
    N_CLASS = config["params"]["N_CLASS"]
    LOSS_FUNCTION = config["params"]["LOSS_FUNCTION"]
    OPTIMIZER = config["params"]["OPTIMIZER"]
    METRICS = config["params"]["METRICS"]

    mlp_obj = MLPerceptron(input_shape=INPUT_SHAPE, epochs=EPOCHS, n_class= N_CLASS)
    model_clf = mlp_obj.classifier()
    print(model_clf.summary())
    _ = mlp_obj.model_compile(model_clf,LOSS_FUNCTION, OPTIMIZER, METRICS)
    history = mlp_obj.model_fit(model_clf, X_train, y_train, X_valid, y_valid)

    artifact_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_dir_path = os.path.join(artifact_dir, model_dir)

    create_dirs([artifact_dir, model_dir_path])

    model_name = config["artifacts"]["model_name"]
    model_path = os.path.join(model_dir_path, model_name)
    save_model(model_clf, model_path)

    plot_dir = config["artifacts"]["plots_dir"]
    plot_dir_path = os.path.join(artifact_dir, plot_dir)
    print(plot_dir_path)
    create_dirs([plot_dir_path])
    plot_name = config["artifacts"]["plot_name"]
    plot_file_path = os.path.join(plot_dir_path, plot_name)

    save_plot(history, plot_file_path)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()

    main(parsed_args.config)






    
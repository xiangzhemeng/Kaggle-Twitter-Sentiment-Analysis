import sys
import argparse
import data_loading
import data_preprocessing
import cnn_training
import xgboost_training





def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='parse args from command line')

    parser.add_argument('-m,', "--mode",
                        help='mode select xgboost / cnn / all',
                        required='True',
                        default='xgboost')

    results = parser.parse_args(args)
    return results.mode


if __name__ == "__main__":

    mode = parse_arg(sys.argv[1:])

    if mode == "xgboost":
        print("====== Start from XGBoost Step ======")
        xgboost_training.run_xgboost()

    elif mode == "cnn" :
        print("====== Start from CNN Step ======")
        cnn_training.run_neural_network()
        xgboost_training.run_xgboost()

    elif mode == "all":
        print("====== Start from preprocessing step ====== ")
        data_loading.gen_data()
        data_preprocessing.run_preprocessing()
        cnn_training.run_neural_network()
        xgboost_training.run_xgboost()

    else:
        print("Wrong argument, please execute again with correct -m argument ")
        sys.exit()

    print("====== End of run.py Please go to data/output to get the prediction result ======")
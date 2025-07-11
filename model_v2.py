# Supress Warnings
import os
# 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"    # suppress TF C++ INFO/WARNING
os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "3"    # suppress absl INFO/WARNING

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Imports
from keras import Sequential
import keras
from keras.layers import InputLayer, Dense
from keras.src.optimizers import SGD
from sklearn.metrics import accuracy_score, roc_auc_score

import aux # import aux.py (separate methods for cleaner code)

print("--------------------------------------------------------------------------------")

# Download data from https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module2/L2/diabetes.csv
# and put it into DATASET_PATH
DATASET_PATH = "data/diabetes.csv"
NUM_EPOCHS = 100

if __name__ == '__main__':

    # Load and split data
    X_train, X_test, y_train, y_test = aux.loadDiabetesData(DATASET_PATH)

    # Preprocessing for the NN
    X_train_norm, X_test_norm = aux.normaliseData(X_train, X_test)
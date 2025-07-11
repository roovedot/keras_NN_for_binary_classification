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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.src.optimizers import SGD

print("--------------------------------------------------------------------------------")

# Download data from https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module2/L2/diabetes.csv
# and put it into DATASET_PATH
DATASET_PATH = "data/diabetes.csv"

def loadDiabetesData(path):

    names = ["times_pregnant", "glucose_tolerance_test", "blood_pressure", "skin_thickness", "insulin", 
         "bmi", "pedigree_function", "age", "has_diabetes"]
    diabetes_df = pd.read_csv(path, names=names, header=0)

    X = diabetes_df.iloc[:, :-1].values
    y = diabetes_df["has_diabetes"].values

    # Split the data to Train, and Test (75%, 25%)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)

    return X_train, X_test, y_train, y_test

def normaliseData(X_train, X_test):
    ## This aids the training of neural nets by providing numerical stability

    normalizer = StandardScaler()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)

    return X_train_norm, X_test_norm


if __name__ == '__main__':
    
    # Load and split data
    X_train, X_test, y_train, y_test = loadDiabetesData(DATASET_PATH)

    # Preprocessing for the NN
    X_train_norm, X_test_norm = normaliseData(X_train, X_test)


    # BUILD AND COMPILE MODEL

    # Architecture
    model_1 = Sequential([
        InputLayer(shape=(8,)), # 8-feature input
        Dense(units=12, activation='sigmoid'), # hidden dense 12-node layer
        Dense(units=1, activation='sigmoid') # Output layer, binary classification
    ], name="superCoolNN")

    # Visualize model   
    print(model_1.summary())
    
    # Compile model
    model_1.compile(
        optimizer=SGD(learning_rate= .003), # SGD Optimizer
        loss="binary_crossentropy", # loss function
        metrics=["accuracy"], # eval metrics
    )
    #print(model_1.compiled)#debug

    # fit and save the run history (returned by the fit function)
    run_hist_1 = model_1.fit(
        X_train_norm, # x
        y_train, # y
        validation_data=(X_test_norm, y_test)) 
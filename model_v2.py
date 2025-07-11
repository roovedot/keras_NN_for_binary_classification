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
from keras.layers import InputLayer, Dense, LeakyReLU, BatchNormalization, Activation, Dropout
from keras.src.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.callbacks import ReduceLROnPlateau

import aux # import aux.py (separate methods for cleaner code)

print("--------------------------------------------------------------------------------")

# Download data from https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module2/L2/diabetes.csv
# and put it into DATASET_PATH
DATASET_PATH = "data/diabetes.csv"
NUM_EPOCHS = 600
LR = 0.003
OPTIMIZER_OPTIONS = [
    SGD(learning_rate=LR), # 0
    Adam(learning_rate=LR) # 1
    ]
OPTIMIZER = OPTIMIZER_OPTIONS[0]

# LR Scheduler helps prevent overfitting
scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1,
)

if __name__ == '__main__':

    # Load and split data
    X_train, X_test, y_train, y_test = aux.loadDiabetesData(DATASET_PATH)

    # Preprocessing for the NN
    X_train_norm, X_test_norm = aux.normaliseData(X_train, X_test)


    # MODEL

    # Architecture
    model_v2 = Sequential([
        InputLayer(shape=(8,)),

        Dense(12), 
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3), # Forces Each neuron not to depend too much on the whole network, and learn a filter that is useful by itself


        Dense(6), 
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid") # Binary classification final activation
    ])

    # Compile
    model_v2.compile(
        optimizer=OPTIMIZER,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Visualize model
    model_v2.summary()

    # Train
    run_hist_2 = model_v2.fit(
        X_train_norm, y_train, 
        validation_data=(X_test_norm, y_test), 
        epochs=NUM_EPOCHS,
        callbacks=[scheduler])

    # Predict
    pred_prob_2 = model_v2.predict(X_test_norm)
    pred_binClass_2 = (pred_prob_2 > 0.5).astype(int)


    # VISUALIZATION

    # Print model performance and plot the roc curve
    print('accuracy is {:.3f}'.format(accuracy_score(y_test,pred_binClass_2)))
    print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,pred_prob_2)))
    aux.plot_roc(y_test, pred_prob_2, 'NN', save_path="data/rocCurve_v2.png") 

    aux.plotHistory(run_hist_2, save_path="data/history_v2.png")
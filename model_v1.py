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
NUM_EPOCHS = 900

if __name__ == '__main__':

    # Load and split data
    X_train, X_test, y_train, y_test = aux.loadDiabetesData(DATASET_PATH)

    # Preprocessing for the NN
    X_train_norm, X_test_norm = aux.normaliseData(X_train, X_test)


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
    # Prints something like this for each epoch:
    #   Epoch 1/200
    #   18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.6538 - loss: 0.6510 - val_accuracy: 0.6406 - val_loss: 0.6660
    run_hist_1 = model_1.fit(
        X_train_norm, # x
        y_train, # y
        validation_data=(X_test_norm, y_test), # Just for the metrics, doesn't affect training
        epochs=NUM_EPOCHS) 
    
    pred_prob_1 = model_1.predict(X_test_norm) # Predict probability of diabetes
    pred_binClass_1 = (pred_prob_1 > 0.5).astype(int) # binary classification, label as diabetes if probability > 50%

    # See predicted probabilities for the 10 first examples
    print(pred_prob_1[:10])


    # VISUALIZATION

    # Print model performance and plot the roc curve
    print('accuracy is {:.3f}'.format(accuracy_score(y_test,pred_binClass_1)))
    print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,pred_prob_1)))
    aux.plot_roc(y_test, pred_prob_1, 'NN', save_path="data/rocCurve.png") 

    aux.plotHistory(run_hist_1, save_path="data/history.png")
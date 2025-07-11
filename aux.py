from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def plot_roc(y_test, y_pred, model_name, save_path = None):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    else:
        plt.show()               # this pops the window
        input("Press ⏎ to exit")  # script blocks here until you hit Enter

def plotHistory(hist, save_path = None):

    fig, ax = plt.subplots()
    ax.plot(hist.history["loss"],'r', marker='.', label="Train Loss")
    ax.plot(hist.history["val_loss"],'b', marker='.', label="Validation Loss")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    else:
        plt.show()               # this pops the window
        input("Press ⏎ to exit")  # script blocks here until you hit Enter
mport numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection
import train_test_split from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report


dataset = pd.read_csv('dataset.csv')
x_complete = dataset.iloc[:, 0:k].values
x = dataset.iloc[:, 1:(k−1)].values
y = dataset.iloc[:, k].values

RF_model = RandomForestClassifier()
RF_model.fit(x_train, y_train)
RF_y_pred = RF_model.predict(x_test)
joblib.dump(RF_model, "RF_model.joblib")
XGB_model = XGBClassifier(use_label_encoder=False)
XGB_model.fit(x_train, y_train)
XGB_y_pred = XGB_model.predict(x_test)
joblib.dump(XGB_model, "XGB_model.joblib")

RF_model = joblib.load("RF_model.joblib")
XGB_model = joblib.load("XGB_model.joblib")

def neural_network(x_train, y_train, x_test, y_test, threshold):
    NN_model = Sequential()
    NN_model.add(Dense(64, activation='relu', input_dim=k))
    NN_model.add(Dense(32, activation='relu'))
    NN_model.add(Dense(1, activation='sigmoid'))
    NN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = NN_model.fit(x_train, y_train, batch_size=8, epochs=60, validation_split=0.2, shuffle=True)
    NN_model.save('NN_model.h5')
    global NN_y_pred_bool
    NN_y_pred = NN_model.predict(x_test)
    NN_y_pred_bool = (NN_y_pred > threshold)
    cm = confusion_matrix(y_test, NN_y_pred_bool)
    print(cm)
    print("Classification report\n%s\n" % (classification_report(y_test, NN_y_pred_bool)))
    print("total frauds in test set ",
    "\n\nStats for Neural Network",
    ":\ncounter of alert ", cm[0][0],
    "\ncatched frauds ", cm[1][1],
    "\nfalse positive ", cm[1][0],
    "\nmissed frauds ", cm[0][1],
    "\nF1 − Score ", f1_score(y_test, NN_y_pred_bool, average='macro'), "\nMCC ", matthews_corrcoef(y_test, NN_y_pred_bool))
    plot_accuracy_loss(history)

def plot_accuracy_loss (history):
    # summarize history for accuracy plt.plot(history.history['accuracy']) plt.plot(history.history['val_accuracy']) plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def stats_and_featureplot(y_pred, y_test, model, model_name):
    counter = 0
    catched_fraud = 0
    false_positive = 0
    global total_fraud
    total_fraud = 0
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy ", model_name, " : %.2f%%" % (accuracy ∗ 100.0))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    for i in range(len(y_test)):
        if y_test[i] == 1:
            total_fraud = total_fraud + 1
    for j in range(len(y_pred)):
        if y_pred[j] > 0:
            counter = counter + 1
            if y_test[j] == 1:
                catched_fraud = catched_fraud + 1
            else:
                false_positive = false_positive + 1
    missed_fraud = total_fraud − catched_fraud
    print("total frauds in test set ", total_fraud,
    "\n\nStats for ", model_name,
    ":\ncounter of alert ", counter,
    "\ncatched frauds ", catched_fraud,
    "\nfalse positive ", false_positive,
    "\nmissed frauds ", missed_fraud,
    "\nF1 − Score ", f1_score(y_test, y_pred, average='macro'), "\nMCC ", matthews_corrcoef(y_test, y_pred))
    title = model_name + ' feature importance'
    feature_names = dataset.columns[1:k]
    plot_feature_importance(model.feature_importances_, feature_names, title)


def plot_feature_importance(importance, names, title): #Create arrays from feature importance and feature names feature_importance = np.array(importance) feature_names = np.array(names)
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names']) #Add chart labels
    plt.title(title)
    plt.xlabel('FEATURE IMPORTANCE') plt.ylabel('FEATURE NAMES')
    plt.show()


def score(XGB_y_pred, RF_y_pred, NN_y_pred_bool, y_test):
    # need to create another function for that
    score_file = pd.DataFrame(columns=['RF', 'XGB', 'NN', 'Score', 'Fraud'])
    highCounter = 0
    mediumCounter = 0
    lowCounter = 0
    highFraud = 0
    mediumFraud = 0
    lowFraud = 0
    highFP = 0
    mediumFP = 0
    lowFP = 0
    missedGeneral = 0
    for i in range(len(y_test)):
        if XGB_y_pred[i] == 1 and RF_y_pred[i] == 1 and NN_y_pred_bool[i] == 1:
            score = 3
            highCounter = highCounter + 1
            if y_test[i] == 1:
                highFraud = highFraud + 1
            else:
                highFP = highFP + 1
        elif ((XGB_y_pred[i] == 1 and RF_y_pred[i] == 1) or (RF_y_pred[i] == 1 and NN_y_pred_bool[i] == 1) or (XGB_y_pred[i] == 1 and NN_y_pred_bool[i] == 1)):
            score = 2
            mediumCounter = mediumCounter + 1
            if y_test[i] == 1:
                mediumFraud = mediumFraud + 1
            else:
                mediumFP = mediumFP + 1
        elif XGB_y_pred[i] == 1 or RF_y_pred[i] == 1 or NN_y_pred_bool[i] == 1:
            score = 1
            lowCounter = lowCounter + 1
            if y_test[i] == 1:
                lowFraud = lowFraud + 1
            else:
                lowFP = lowFP + 1
        else:
            score = 0
        missedGeneral = total_fraud − highFraud − mediumFraud − lowFraud
        score_file.loc[i] = [RF_y_pred[i], XGB_y_pred[i], NN_y_pred_bool[i], score, y_test[i]]
    print("\n\ntotal frauds in test set ", total_fraud, "\n\nStats for High Score:", "\ncounter of High Score ", highCounter, "\ncatched frauds ", highFraud, "\nfalse positive ", highFP, "\n\nStats for Medium Score:", "\ncounter of alert ", mediumCounter, "\ncatched frauds ", mediumFraud, "\nfalse positive ", mediumFP, "\n\nStats for Low Score:", "\ncounter of alert ", lowCounter, "\ncatched frauds ", lowFraud, "\nfalse positive ", lowFP, "\n\nmissed frauds in general", missedGeneral)
    score_file.to_csv('ScoreFile.csv')
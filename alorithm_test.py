# -*- coding: utf-8 -*-

"""
KAMILLA HEIMAR ANDERSEN
kahean@build.aau.dk
"""

#%%
##############################################################################
### PACKAGES ###
##############################################################################

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
import os
import time
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report

#%%
##############################################################################
### DEFINE LOCATION ###
##############################################################################

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

os.chdir(dname)
current_working_folder = os.getcwd()
print(current_working_folder)

#%%
##############################################################################
### LOAD DATA ###
##############################################################################

dataset = pd.read_csv('your_location.csv')
dataset['DateTime'] = pd.to_datetime(dataset['DateTime'], format = '%d-%m-%y %H:%M')

#%%

# make a copy of the dataset to remove the datetime column
full_dataset = dataset.copy()

#%%
##############################################################################
### DIVID THE DATA INTO SPLITS BASED ON DAY_LABEL ###
##############################################################################

X = full_dataset[['co2_concentration', 'air_temperature', 'relative_humidity', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'kitchen', 'livingroom', 'bedroom', 'office', 'kitchen_livingroom']]
y_full_dataset = full_dataset['occupancy_ground_truth']

#%%
##############################################################################
### CYCLIC ENCODING ###
##############################################################################

# make two functions that performs cyclic encoding
def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# apply the cyclic encoding
X["day_sin"] = sin_transformer(7).fit_transform(X)["Day number of the week"]
X["day_cos"] = cos_transformer(7).fit_transform(X)["Day number of the week"]

X["hour_sin"] = sin_transformer(24).fit_transform(X)["Hour of the day"]
X["hour_cos"] = cos_transformer(24).fit_transform(X)["Hour of the day"]

# plot figure
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
X[["day_sin", "day_cos"]].plot(ax=ax[0])
X[["hour_sin", "hour_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation");

#%%
##############################################################################
### SOME MORE DATA TREATMENT ###
##############################################################################

# make a copy of the day label
room_label = X['room_no'].copy()

# create groups for the cv split
groups = room_label.values
X.drop(['room_no'], axis = 1, inplace = True)

# define the input as a copy of the x df
X_full_dataset = X.copy()

# remove the day number and hour of the day
X_full_dataset.drop(['Day number of the week'], axis = 1, inplace = True)
X_full_dataset.drop(['Hour of the day'], axis = 1, inplace = True)
X_full_dataset.drop(['day_label'], axis = 1, inplace = True)
X_full_dataset.drop(['room_type'], axis = 1, inplace = True)
X_full_dataset.drop(['floor_area'], axis = 1, inplace = True)

#%%
#%%
##############################################################################
### LOGISTIC REGRESSION ###
##############################################################################
### LOGISTIC REGRESSION ###
##############################################################################

# Start the timer
start_time = time.time()

"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

bas = []  # for balanced accuracy
accs = [] # for accuracy
mccs = [] # for Matthew's Correlation Coefficient
briers = [] # for Brier Score Loss
roc_aucs = [] # for AUC-ROC
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall

# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_mcc = 0
total_brier = 0
total_roc_auc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0 
total_cm = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix with zeros

# Number of folds
n_folds = cross_validation.get_n_splits()

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    X_model_selection = X_full_dataset.iloc[model_selection_index]
    X_model_evaluation = X_full_dataset.iloc[model_evaluation_index]
    y_model_selection = y_full_dataset.iloc[model_selection_index]
    y_model_evaluation = y_full_dataset.iloc[model_evaluation_index]

    # print these dataframes
    print(f"X_model_selection for fold {i}:\n{X_model_selection}")
    print(f"X_model_evaluation for fold {i}:\n{X_model_evaluation}")
    print(f"y_model_selection for fold {i}:\n{y_model_selection}")
    print(f"y_model_evaluation for fold {i}:\n{y_model_evaluation}")
    
    # create and fit logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_model_selection, y_model_selection)

    # make predictions
    y_pred = logreg.predict(X_model_evaluation)
    y_pred_proba = logreg.predict_proba(X_model_evaluation)[:, 1]

    # compute performance metrics
    ba = balanced_accuracy_score(y_model_evaluation, y_pred)
    acc = accuracy_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    brier = brier_score_loss(y_model_evaluation, y_pred_proba)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred_proba)
    f1 = f1_score(y_model_evaluation, y_pred)
    cm = confusion_matrix(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)  
    recall = recall_score(y_model_evaluation, y_pred)
      
    # print classification report
    print(f"Classification Report for fold {i}:")
    print(classification_report(y_model_evaluation, y_pred))

    # update running total of each metric
    total_ba += ba
    total_acc += acc
    total_mcc += mcc
    total_brier += brier
    total_roc_auc += roc_auc
    total_f1 += f1
    total_precision += precision  # Update total Precision
    total_recall += recall  # Update total Recall
    total_cm += cm

    bas.append(ba)
    accs.append(acc)
    mccs.append(mcc)
    briers.append(brier)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

# compute average of each metric
avg_ba = total_ba / n_folds
avg_acc = total_acc / n_folds
avg_mcc = total_mcc / n_folds
avg_brier = total_brier / n_folds
avg_roc_auc = total_roc_auc / n_folds
avg_f1 = total_f1 / n_folds
avg_precision = total_precision / n_folds  # Compute average Precision
avg_recall = total_recall / n_folds  # Compute average Recall

std_ba = np.std(bas)
std_acc = np.std(accs)
std_mcc = np.std(mccs)
std_brier = np.std(briers)
std_roc_auc = np.std(roc_aucs)
std_f1 = np.std(f1s)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
   
# print averages
print("Average Balanced Accuracy:", avg_ba)
print("Average Accuracy:", avg_acc)
print("Average Matthew's Correlation Coefficient:", avg_mcc)
print("Average Brier Score Loss:", avg_brier)
print("Average AUC-ROC:", avg_roc_auc)
print("Average F1 Score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)  
print("Total Confusion Matrix:\n", total_cm)

print("Standard Deviation of Balanced Accuracy:", std_ba)
print("Standard Deviation of Accuracy:", std_acc)
print("Standard Deviation of Matthew's Correlation Coefficient:", std_mcc)
print("Standard Deviation of Brier Score Loss:", std_brier)
print("Standard Deviation of AUC-ROC:", std_roc_auc)
print("Standard Deviation of F1 Score:", std_f1)
print("Standard Deviation of Precision:", std_precision)
print("Standard Deviation of Recall:", std_recall)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))
    
#%%
##############################################################################
### SUPPORT VECTOR MACHINE ###
##############################################################################

# Start the timer
start_time = time.time()

"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

from sklearn.svm import SVC

bas = []  # for balanced accuracy
accs = [] # for accuracy
mccs = [] # for Matthew's Correlation Coefficient
briers = [] # for Brier Score Loss
roc_aucs = [] # for AUC-ROC
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall

# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_mcc = 0
total_brier = 0
total_roc_auc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0 
total_cm = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix with zeros

# Number of folds
n_folds = cross_validation.get_n_splits()

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    X_model_selection = X_full_dataset.iloc[model_selection_index]
    X_model_evaluation = X_full_dataset.iloc[model_evaluation_index]
    y_model_selection = y_full_dataset.iloc[model_selection_index]
    y_model_evaluation = y_full_dataset.iloc[model_evaluation_index]

    # print these dataframes
    print(f"X_model_selection for fold {i}:\n{X_model_selection}")
    print(f"X_model_evaluation for fold {i}:\n{X_model_evaluation}")
    print(f"y_model_selection for fold {i}:\n{y_model_selection}")
    print(f"y_model_evaluation for fold {i}:\n{y_model_evaluation}")
    
    # create and fit SVM model (probability=True is needed for predict_proba)
    svm = SVC(probability=True)
    svm.fit(X_model_selection, y_model_selection)

    # make predictions
    y_pred = svm.predict(X_model_evaluation)
    y_pred_proba = svm.predict_proba(X_model_evaluation)[:, 1]

    # compute performance metrics
    ba = balanced_accuracy_score(y_model_evaluation, y_pred)
    acc = accuracy_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    brier = brier_score_loss(y_model_evaluation, y_pred_proba)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred_proba)
    f1 = f1_score(y_model_evaluation, y_pred)
    cm = confusion_matrix(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)  
    recall = recall_score(y_model_evaluation, y_pred)
      
    # print classification report
    print(f"Classification Report for fold {i}:")
    print(classification_report(y_model_evaluation, y_pred))

    # update running total of each metric
    total_ba += ba
    total_acc += acc
    total_mcc += mcc
    total_brier += brier
    total_roc_auc += roc_auc
    total_f1 += f1
    total_precision += precision  # Update total Precision
    total_recall += recall  # Update total Recall
    total_cm += cm

    bas.append(ba)
    accs.append(acc)
    mccs.append(mcc)
    briers.append(brier)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

# compute average of each metric
avg_ba = total_ba / n_folds
avg_acc = total_acc / n_folds
avg_mcc = total_mcc / n_folds
avg_brier = total_brier / n_folds
avg_roc_auc = total_roc_auc / n_folds
avg_f1 = total_f1 / n_folds
avg_precision = total_precision / n_folds  # Compute average Precision
avg_recall = total_recall / n_folds  # Compute average Recall

std_ba = np.std(bas)
std_acc = np.std(accs)
std_mcc = np.std(mccs)
std_brier = np.std(briers)
std_roc_auc = np.std(roc_aucs)
std_f1 = np.std(f1s)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
   
# print averages
print("Average Balanced Accuracy:", avg_ba)
print("Average Accuracy:", avg_acc)
print("Average Matthew's Correlation Coefficient:", avg_mcc)
print("Average Brier Score Loss:", avg_brier)
print("Average AUC-ROC:", avg_roc_auc)
print("Average F1 Score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)  
print("Total Confusion Matrix:\n", total_cm)

print("Standard Deviation of Balanced Accuracy:", std_ba)
print("Standard Deviation of Accuracy:", std_acc)
print("Standard Deviation of Matthew's Correlation Coefficient:", std_mcc)
print("Standard Deviation of Brier Score Loss:", std_brier)
print("Standard Deviation of AUC-ROC:", std_roc_auc)
print("Standard Deviation of F1 Score:", std_f1)
print("Standard Deviation of Precision:", std_precision)
print("Standard Deviation of Recall:", std_recall)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))

#%%
##############################################################################
### RANDOM FOREST ###
##############################################################################

# Start the timer
start_time = time.time()

"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

from sklearn.ensemble import RandomForestClassifier

bas = []  # for balanced accuracy
accs = [] # for accuracy
mccs = [] # for Matthew's Correlation Coefficient
briers = [] # for Brier Score Loss
roc_aucs = [] # for AUC-ROC
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall

# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_mcc = 0
total_brier = 0
total_roc_auc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0
total_cm = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix with zeros

# Number of folds
n_folds = cross_validation.get_n_splits()

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    X_model_selection = X_full_dataset.iloc[model_selection_index]
    X_model_evaluation = X_full_dataset.iloc[model_evaluation_index]
    y_model_selection = y_full_dataset.iloc[model_selection_index]
    y_model_evaluation = y_full_dataset.iloc[model_evaluation_index]

    # print these dataframes
    print(f"X_model_selection for fold {i}:\n{X_model_selection}")
    print(f"X_model_evaluation for fold {i}:\n{X_model_evaluation}")
    print(f"y_model_selection for fold {i}:\n{y_model_selection}")
    print(f"y_model_evaluation for fold {i}:\n{y_model_evaluation}")
    
    # create and fit Random Forest model
    rf = RandomForestClassifier()
    rf.fit(X_model_selection, y_model_selection)

    # make predictions
    y_pred = rf.predict(X_model_evaluation)
    y_pred_proba = rf.predict_proba(X_model_evaluation)[:, 1]

    # compute performance metrics
    ba = balanced_accuracy_score(y_model_evaluation, y_pred)
    acc = accuracy_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    brier = brier_score_loss(y_model_evaluation, y_pred_proba)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred_proba)
    f1 = f1_score(y_model_evaluation, y_pred)
    cm = confusion_matrix(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)  
    recall = recall_score(y_model_evaluation, y_pred)
      
    # print classification report
    print(f"Classification Report for fold {i}:")
    print(classification_report(y_model_evaluation, y_pred))

    # update running total of each metric
    total_ba += ba
    total_acc += acc
    total_mcc += mcc
    total_brier += brier
    total_roc_auc += roc_auc
    total_f1 += f1
    total_precision += precision  # Update total Precision
    total_recall += recall  # Update total Recall
    total_cm += cm

    bas.append(ba)
    accs.append(acc)
    mccs.append(mcc)
    briers.append(brier)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

# compute average of each metric
avg_ba = total_ba / n_folds
avg_acc = total_acc / n_folds
avg_mcc = total_mcc / n_folds
avg_brier = total_brier / n_folds
avg_roc_auc = total_roc_auc / n_folds
avg_f1 = total_f1 / n_folds
avg_precision = total_precision / n_folds  # Compute average Precision
avg_recall = total_recall / n_folds  # Compute average Recall

std_ba = np.std(bas)
std_acc = np.std(accs)
std_mcc = np.std(mccs)
std_brier = np.std(briers)
std_roc_auc = np.std(roc_aucs)
std_f1 = np.std(f1s)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
   
# print averages
print("Average Balanced Accuracy:", avg_ba)
print("Average Accuracy:", avg_acc)
print("Average Matthew's Correlation Coefficient:", avg_mcc)
print("Average Brier Score Loss:", avg_brier)
print("Average AUC-ROC:", avg_roc_auc)
print("Average F1 Score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)  
print("Total Confusion Matrix:\n", total_cm)

print("Standard Deviation of Balanced Accuracy:", std_ba)
print("Standard Deviation of Accuracy:", std_acc)
print("Standard Deviation of Matthew's Correlation Coefficient:", std_mcc)
print("Standard Deviation of Brier Score Loss:", std_brier)
print("Standard Deviation of AUC-ROC:", std_roc_auc)
print("Standard Deviation of F1 Score:", std_f1)
print("Standard Deviation of Precision:", std_precision)
print("Standard Deviation of Recall:", std_recall)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))

#%%
##############################################################################
### NAIVE BAYES ###
##############################################################################

# Start the timer
start_time = time.time()

"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

from sklearn.naive_bayes import GaussianNB

bas = []  # for balanced accuracy
accs = [] # for accuracy
mccs = [] # for Matthew's Correlation Coefficient
briers = [] # for Brier Score Loss
roc_aucs = [] # for AUC-ROC
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall

# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_mcc = 0
total_brier = 0
total_roc_auc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0 
total_cm = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix with zeros

# Number of folds
n_folds = cross_validation.get_n_splits()

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    X_model_selection = X_full_dataset.iloc[model_selection_index]
    X_model_evaluation = X_full_dataset.iloc[model_evaluation_index]
    y_model_selection = y_full_dataset.iloc[model_selection_index]
    y_model_evaluation = y_full_dataset.iloc[model_evaluation_index]

    # print these dataframes
    print(f"X_model_selection for fold {i}:\n{X_model_selection}")
    print(f"X_model_evaluation for fold {i}:\n{X_model_evaluation}")
    print(f"y_model_selection for fold {i}:\n{y_model_selection}")
    print(f"y_model_evaluation for fold {i}:\n{y_model_evaluation}")
    
    # create and fit Naive Bayes model
    nb = GaussianNB()
    nb.fit(X_model_selection, y_model_selection)

    # make predictions
    y_pred = nb.predict(X_model_evaluation)
    y_pred_proba = nb.predict_proba(X_model_evaluation)[:, 1]

    # compute performance metrics
    ba = balanced_accuracy_score(y_model_evaluation, y_pred)
    acc = accuracy_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    brier = brier_score_loss(y_model_evaluation, y_pred_proba)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred_proba)
    f1 = f1_score(y_model_evaluation, y_pred)
    cm = confusion_matrix(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)  
    recall = recall_score(y_model_evaluation, y_pred)
      
    # print classification report
    print(f"Classification Report for fold {i}:")
    print(classification_report(y_model_evaluation, y_pred))

    # update running total of each metric
    total_ba += ba
    total_acc += acc
    total_mcc += mcc
    total_brier += brier
    total_roc_auc += roc_auc
    total_f1 += f1
    total_precision += precision  # Update total Precision
    total_recall += recall  # Update total Recall
    total_cm += cm

    bas.append(ba)
    accs.append(acc)
    mccs.append(mcc)
    briers.append(brier)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

# compute average of each metric
avg_ba = total_ba / n_folds
avg_acc = total_acc / n_folds
avg_mcc = total_mcc / n_folds
avg_brier = total_brier / n_folds
avg_roc_auc = total_roc_auc / n_folds
avg_f1 = total_f1 / n_folds
avg_precision = total_precision / n_folds  # Compute average Precision
avg_recall = total_recall / n_folds  # Compute average Recall

std_ba = np.std(bas)
std_acc = np.std(accs)
std_mcc = np.std(mccs)
std_brier = np.std(briers)
std_roc_auc = np.std(roc_aucs)
std_f1 = np.std(f1s)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
   
# print averages
print("Average Balanced Accuracy:", avg_ba)
print("Average Accuracy:", avg_acc)
print("Average Matthew's Correlation Coefficient:", avg_mcc)
print("Average Brier Score Loss:", avg_brier)
print("Average AUC-ROC:", avg_roc_auc)
print("Average F1 Score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)  
print("Total Confusion Matrix:\n", total_cm)

print("Standard Deviation of Balanced Accuracy:", std_ba)
print("Standard Deviation of Accuracy:", std_acc)
print("Standard Deviation of Matthew's Correlation Coefficient:", std_mcc)
print("Standard Deviation of Brier Score Loss:", std_brier)
print("Standard Deviation of AUC-ROC:", std_roc_auc)
print("Standard Deviation of F1 Score:", std_f1)
print("Standard Deviation of Precision:", std_precision)
print("Standard Deviation of Recall:", std_recall)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))

#%%
##############################################################################
### KNN ###
##############################################################################

# Start the timer
start_time = time.time()

"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

from sklearn.neighbors import KNeighborsClassifier

bas = []  # for balanced accuracy
accs = [] # for accuracy
mccs = [] # for Matthew's Correlation Coefficient
briers = [] # for Brier Score Loss
roc_aucs = [] # for AUC-ROC
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall

# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_mcc = 0
total_brier = 0
total_roc_auc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0 
total_cm = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix with zeros

# Number of folds
n_folds = cross_validation.get_n_splits()

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    X_model_selection = X_full_dataset.iloc[model_selection_index]
    X_model_evaluation = X_full_dataset.iloc[model_evaluation_index]
    y_model_selection = y_full_dataset.iloc[model_selection_index]
    y_model_evaluation = y_full_dataset.iloc[model_evaluation_index]

    # print these dataframes
    print(f"X_model_selection for fold {i}:\n{X_model_selection}")
    print(f"X_model_evaluation for fold {i}:\n{X_model_evaluation}")
    print(f"y_model_selection for fold {i}:\n{y_model_selection}")
    print(f"y_model_evaluation for fold {i}:\n{y_model_evaluation}")
    
    # create and fit KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_model_selection, y_model_selection)

    # make predictions
    y_pred = knn.predict(X_model_evaluation)
    y_pred_proba = knn.predict_proba(X_model_evaluation)[:, 1]

    # compute performance metrics
    ba = balanced_accuracy_score(y_model_evaluation, y_pred)
    acc = accuracy_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    brier = brier_score_loss(y_model_evaluation, y_pred_proba)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred_proba)
    f1 = f1_score(y_model_evaluation, y_pred)
    cm = confusion_matrix(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)  
    recall = recall_score(y_model_evaluation, y_pred)
      
    # print classification report
    print(f"Classification Report for fold {i}:")
    print(classification_report(y_model_evaluation, y_pred))

    # update running total of each metric
    total_ba += ba
    total_acc += acc
    total_mcc += mcc
    total_brier += brier
    total_roc_auc += roc_auc
    total_f1 += f1
    total_precision += precision  # Update total Precision
    total_recall += recall  # Update total Recall
    total_cm += cm

    bas.append(ba)
    accs.append(acc)
    mccs.append(mcc)
    briers.append(brier)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

# compute average of each metric
avg_ba = total_ba / n_folds
avg_acc = total_acc / n_folds
avg_mcc = total_mcc / n_folds
avg_brier = total_brier / n_folds
avg_roc_auc = total_roc_auc / n_folds
avg_f1 = total_f1 / n_folds
avg_precision = total_precision / n_folds  # Compute average Precision
avg_recall = total_recall / n_folds  # Compute average Recall

std_ba = np.std(bas)
std_acc = np.std(accs)
std_mcc = np.std(mccs)
std_brier = np.std(briers)
std_roc_auc = np.std(roc_aucs)
std_f1 = np.std(f1s)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
   
# print averages
print("Average Balanced Accuracy:", avg_ba)
print("Average Accuracy:", avg_acc)
print("Average Matthew's Correlation Coefficient:", avg_mcc)
print("Average Brier Score Loss:", avg_brier)
print("Average AUC-ROC:", avg_roc_auc)
print("Average F1 Score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)  
print("Total Confusion Matrix:\n", total_cm)

print("Standard Deviation of Balanced Accuracy:", std_ba)
print("Standard Deviation of Accuracy:", std_acc)
print("Standard Deviation of Matthew's Correlation Coefficient:", std_mcc)
print("Standard Deviation of Brier Score Loss:", std_brier)
print("Standard Deviation of AUC-ROC:", std_roc_auc)
print("Standard Deviation of F1 Score:", std_f1)
print("Standard Deviation of Precision:", std_precision)
print("Standard Deviation of Recall:", std_recall)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))

#%%
##############################################################################
### XG-BOOST ###
##############################################################################

# Start the timer
start_time = time.time()
    
"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

bas = []  # for balanced accuracy
accs = [] # for accuracy
mccs = [] # for Matthew's Correlation Coefficient
briers = [] # for Brier Score Loss
roc_aucs = [] # for AUC-ROC
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall    
    
# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_mcc = 0
total_brier = 0
total_roc_auc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0  
total_cm = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix with zeros

# Number of folds
n_folds = cross_validation.get_n_splits()

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_full_dataset, y_full_dataset, groups = groups)):
    X_model_selection = X_full_dataset.iloc[model_selection_index]
    X_model_evaluation = X_full_dataset.iloc[model_evaluation_index]
    y_model_selection = y_full_dataset.iloc[model_selection_index]
    y_model_evaluation = y_full_dataset.iloc[model_evaluation_index]

    # print these dataframes
    print(f"X_model_selection for fold {i}:\n{X_model_selection}")
    print(f"X_model_evaluation for fold {i}:\n{X_model_evaluation}")
    print(f"y_model_selection for fold {i}:\n{y_model_selection}")
    print(f"y_model_evaluation for fold {i}:\n{y_model_evaluation}")
    
    # create and fit XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False)
    xgb_model.fit(X_model_selection, y_model_selection)

    # make predictions
    y_pred = xgb_model.predict(X_model_evaluation)
    y_pred_proba = xgb_model.predict_proba(X_model_evaluation)[:, 1]

    # compute performance metrics
    ba = balanced_accuracy_score(y_model_evaluation, y_pred)
    acc = accuracy_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    brier = brier_score_loss(y_model_evaluation, y_pred_proba)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred_proba)
    f1 = f1_score(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)  # Compute Precision
    recall = recall_score(y_model_evaluation, y_pred)  # Compute Recall
    cm = confusion_matrix(y_model_evaluation, y_pred)
    
    # print classification report
    print(f"Classification Report for fold {i}:")
    print(classification_report(y_model_evaluation, y_pred))

    # update running total of each metric
    total_ba += ba
    total_acc += acc
    total_mcc += mcc
    total_brier += brier
    total_roc_auc += roc_auc
    total_f1 += f1
    total_precision += precision  # Update total Precision
    total_recall += recall  # Update total Recall
    total_cm += cm

    bas.append(ba)
    accs.append(acc)
    mccs.append(mcc)
    briers.append(brier)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

# compute average of each metric
avg_ba = total_ba / n_folds
avg_acc = total_acc / n_folds
avg_mcc = total_mcc / n_folds
avg_brier = total_brier / n_folds
avg_roc_auc = total_roc_auc / n_folds
avg_f1 = total_f1 / n_folds
avg_precision = total_precision / n_folds  # Compute average Precision
avg_recall = total_recall / n_folds  # Compute average Recall

std_ba = np.std(bas)
std_acc = np.std(accs)
std_mcc = np.std(mccs)
std_brier = np.std(briers)
std_roc_auc = np.std(roc_aucs)
std_f1 = np.std(f1s)
std_precision = np.std(precisions)
std_recall = np.std(recalls)
   
# print averages
print("Average Balanced Accuracy:", avg_ba)
print("Average Accuracy:", avg_acc)
print("Average Matthew's Correlation Coefficient:", avg_mcc)
print("Average Brier Score Loss:", avg_brier)
print("Average AUC-ROC:", avg_roc_auc)
print("Average F1 Score:", avg_f1)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)  
print("Total Confusion Matrix:\n", total_cm)

print("Standard Deviation of Balanced Accuracy:", std_ba)
print("Standard Deviation of Accuracy:", std_acc)
print("Standard Deviation of Matthew's Correlation Coefficient:", std_mcc)
print("Standard Deviation of Brier Score Loss:", std_brier)
print("Standard Deviation of AUC-ROC:", std_roc_auc)
print("Standard Deviation of F1 Score:", std_f1)
print("Standard Deviation of Precision:", std_precision)
print("Standard Deviation of Recall:", std_recall)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))

#%%
##############################################################################
### SCRIPT END ###
##############################################################################
# -*- coding: utf-8 -*-

"""
KAMILLA HEIMAR ANDERSEN
kahean@build.aau.dk
"""

'''
The type of cross validation, number of folds and data frames for room type needs 
to be manually changed accordingly to the description in the following publications: XXX
to obtain the same results. 
'''


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

"use previous dataset"

X = full_dataset[['co2_concentration', 'air_temperature', 'relative_humidity', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'kitchen', 'livingroom', 'bedroom', 'office', 'kitchen_livingroom']]
y_full_dataset = full_dataset['occupancy_ground_truth']

#%%

"create separate dataframes based on room_type"

df_room1 = full_dataset[full_dataset['room_type'] == 1]
df_room2 = full_dataset[full_dataset['room_type'] == 2]
df_room3 = full_dataset[full_dataset['room_type'] == 3]
df_room4 = full_dataset[full_dataset['room_type'] == 4]
df_room5 = full_dataset[full_dataset['room_type'] == 5]

#%%

# room type 1: bedroom
X_1 = df_room1[['co2_concentration', 'air_temperature', 'relative_humidity', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'bedroom']]
y_1 = df_room1['occupancy_ground_truth']

def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

X_1["day_sin"] = sin_transformer(7).fit_transform(X_1)["Day number of the week"]
X_1["day_cos"] = cos_transformer(7).fit_transform(X_1)["Day number of the week"]

X_1["hour_sin"] = sin_transformer(24).fit_transform(X_1)["Hour of the day"]
X_1["hour_cos"] = cos_transformer(24).fit_transform(X_1)["Hour of the day"]

#%%

# make a copy of the day label
day_label_1 = X_1['day_label'].copy()

# create groups for the cv split
groups_1 = day_label_1.values
X_1.drop(['day_label'], axis = 1, inplace = True)

# remove the day number and hour of the day
X_1.drop(['Day number of the week'], axis = 1, inplace = True)
X_1.drop(['Hour of the day'], axis = 1, inplace = True)
X_1.drop(['room_type'], axis = 1, inplace = True)
X_1.drop(['room_no'], axis = 1, inplace = True)
X_1.drop(['floor_area'], axis = 1, inplace = True)
X_1.drop(['bedroom'], axis = 1, inplace = True)

#%%

"bedroom"
# Count the number of samples in each class
class_counts = df_room1['occupancy_ground_truth'].value_counts()

# Calculate the class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]

print("Class Distribution:")
print(class_counts)
print("Imbalance Ratio:", imbalance_ratio)

occupied_means = df_room1[df_room1['occupancy_ground_truth'] == 1][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()
unoccupied_means = df_room1[df_room1['occupancy_ground_truth'] == 0][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()

print("Mean values when bedroom is occupied:")
print(occupied_means)
print("\nMean values when bedroom is unoccupied:")
print(unoccupied_means)

#%%

# room type 2: office
X_2 = df_room2[['co2_concentration', 'air_temperature', 'relative_humidity', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'office']]
y_2 = df_room2['occupancy_ground_truth']

def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

X_2["day_sin"] = sin_transformer(7).fit_transform(X_2)["Day number of the week"]
X_2["day_cos"] = cos_transformer(7).fit_transform(X_2)["Day number of the week"]

X_2["hour_sin"] = sin_transformer(24).fit_transform(X_2)["Hour of the day"]
X_2["hour_cos"] = cos_transformer(24).fit_transform(X_2)["Hour of the day"]

# make a copy of the day label
room_label_2 = X_2['room_no'].copy()

# create groups for the cv split
groups_2 = room_label_2.values
X_2.drop(['room_no'], axis = 1, inplace = True)

# remove the day number and hour of the day
X_2.drop(['Day number of the week'], axis = 1, inplace = True)
X_2.drop(['Hour of the day'], axis = 1, inplace = True)
X_2.drop(['room_type'], axis = 1, inplace = True)
X_2.drop(['floor_area'], axis = 1, inplace = True)
X_2.drop(['office'], axis = 1, inplace = True)
X_2.drop(['day_label'], axis = 1, inplace = True)

X_2 = X_2.reset_index(drop=True)
y_2 = y_2.reset_index(drop=True)

#%%

"office"
# Count the number of samples in each class
class_counts = df_room2['occupancy_ground_truth'].value_counts()

# Calculate the class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]

print("Class Distribution:")
print(class_counts)
print("Imbalance Ratio:", imbalance_ratio)

occupied_means = df_room2[df_room2['occupancy_ground_truth'] == 1][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()
unoccupied_means = df_room2[df_room2['occupancy_ground_truth'] == 0][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()

print("Mean values when bedroom is occupied:")
print(occupied_means)
print("\nMean values when bedroom is unoccupied:")
print(unoccupied_means)

#%%

# room type 3: kitchen
X_3 = df_room3[['co2_concentration', 'air_temperature', 'relative_humidity', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'kitchen']]
y_3 = df_room3['occupancy_ground_truth']

def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

X_3["day_sin"] = sin_transformer(7).fit_transform(X_3)["Day number of the week"]
X_3["day_cos"] = cos_transformer(7).fit_transform(X_3)["Day number of the week"]

X_3["hour_sin"] = sin_transformer(24).fit_transform(X_3)["Hour of the day"]
X_3["hour_cos"] = cos_transformer(24).fit_transform(X_3)["Hour of the day"]

# make a copy of the day label
day_label_3 = X_3['day_label'].copy()

# create groups for the cv split
groups_3 = day_label_3.values
X_3.drop(['day_label'], axis = 1, inplace = True)

# remove the day number and hour of the day
X_3.drop(['Day number of the week'], axis = 1, inplace = True)
X_3.drop(['Hour of the day'], axis = 1, inplace = True)
X_3.drop(['room_type'], axis = 1, inplace = True)
X_3.drop(['room_no'], axis = 1, inplace = True)
X_3.drop(['floor_area'], axis = 1, inplace = True)
X_3.drop(['kitchen'], axis = 1, inplace = True)

#%%

"kitchen"
# Count the number of samples in each class
class_counts = df_room3['occupancy_ground_truth'].value_counts()

# Calculate the class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]

print("Class Distribution:")
print(class_counts)
print("Imbalance Ratio:", imbalance_ratio)

occupied_means = df_room3[df_room3['occupancy_ground_truth'] == 1][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()
unoccupied_means = df_room3[df_room3['occupancy_ground_truth'] == 0][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()

print("Mean values when bedroom is occupied:")
print(occupied_means)
print("\nMean values when bedroom is unoccupied:")
print(unoccupied_means)

#%%

# room type 4: living room 
X_4 = df_room4[['co2_concentration', 'air_temperature', 'relative_humidity', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'livingroom']]
y_4 = df_room4['occupancy_ground_truth']

def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

X_4["day_sin"] = sin_transformer(7).fit_transform(X_4)["Day number of the week"]
X_4["day_cos"] = cos_transformer(7).fit_transform(X_4)["Day number of the week"]

X_4["hour_sin"] = sin_transformer(24).fit_transform(X_4)["Hour of the day"]
X_4["hour_cos"] = cos_transformer(24).fit_transform(X_4)["Hour of the day"]

# make a copy of the day label
day_label_4 = X_4['day_label'].copy()

# create groups for the cv split
groups_4 = day_label_4.values
X_4.drop(['day_label'], axis = 1, inplace = True)

# remove the day number and hour of the day
X_4.drop(['Day number of the week'], axis = 1, inplace = True)
X_4.drop(['Hour of the day'], axis = 1, inplace = True)
X_4.drop(['room_type'], axis = 1, inplace = True)
X_4.drop(['room_no'], axis = 1, inplace = True)
X_4.drop(['floor_area'], axis = 1, inplace = True)
X_4.drop(['livingroom'], axis = 1, inplace = True)

#%%

"livingroom"
# Count the number of samples in each class
class_counts = df_room4['occupancy_ground_truth'].value_counts()

# Calculate the class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]

print("Class Distribution:")
print(class_counts)
print("Imbalance Ratio:", imbalance_ratio)

occupied_means = df_room4[df_room4['occupancy_ground_truth'] == 1][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()
unoccupied_means = df_room4[df_room4['occupancy_ground_truth'] == 0][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()

print("Mean values when bedroom is occupied:")
print(occupied_means)
print("\nMean values when bedroom is unoccupied:")
print(unoccupied_means)

#%%

# room type 5: kitchen/living room
X_5 = df_room5[['co2_concentration', 'air_temperature', 'relative_humidity', 'apt_no', 'room_type', 'room_no', 'floor_area', 'Hour of the day', 'Day number of the week', 'day_label', 'kitchen_livingroom']]
y_5 = df_room5['occupancy_ground_truth']

def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

X_5["day_sin"] = sin_transformer(7).fit_transform(X_5)["Day number of the week"]
X_5["day_cos"] = cos_transformer(7).fit_transform(X_5)["Day number of the week"]

X_5["hour_sin"] = sin_transformer(24).fit_transform(X_5)["Hour of the day"]
X_5["hour_cos"] = cos_transformer(24).fit_transform(X_5)["Hour of the day"]

# make a copy of the day label
room_label_5 = X_5['room_no'].copy()

# create groups for the cv split
groups_5 = room_label_5.values
X_5.drop(['room_no'], axis = 1, inplace = True)

# remove the day number and hour of the day
X_5.drop(['Day number of the week'], axis = 1, inplace = True)
X_5.drop(['Hour of the day'], axis = 1, inplace = True)
X_5.drop(['floor_area'], axis = 1, inplace = True)
X_5.drop(['kitchen_livingroom'], axis = 1, inplace = True)
X_5.drop(['day_label'], axis = 1, inplace = True)
X_5.drop(['room_type'], axis = 1, inplace = True)
X_5.drop(['apt_no'], axis = 1, inplace = True)

X_5 = X_5.reset_index(drop=True)
y_5 = y_5.reset_index(drop=True)

#%%

"kitchen_livingroom"
# Count the number of samples in each class
class_counts = df_room5['occupancy_ground_truth'].value_counts()

# Calculate the class imbalance ratio
imbalance_ratio = class_counts[0] / class_counts[1]

print("Class Distribution:")
print(class_counts)
print("Imbalance Ratio:", imbalance_ratio)

occupied_means = df_room5[df_room5['occupancy_ground_truth'] == 1][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()
unoccupied_means = df_room5[df_room5['occupancy_ground_truth'] == 0][['co2_concentration', 'air_temperature', 'relative_humidity']].mean()

print("Mean values when bedroom is occupied:")
print(occupied_means)
print("\nMean values when bedroom is unoccupied:")
print(unoccupied_means)

#%%
##############################################################################
### LOGISTIC REGRESSION ###
##############################################################################

# Start the timer
start_time = time.time()

"Define cross validation split characteristics"
cross_validation = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_5, y_5, groups = groups_5)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups_5[train_index]}")
    print(f"  Test:  index={test_index}, group={groups_5[test_index]}")

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

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_5, y_5, groups = groups_5)):
    X_model_selection = X_5.iloc[model_selection_index]
    X_model_evaluation = X_5.iloc[model_evaluation_index]
    y_model_selection = y_5.iloc[model_selection_index]
    y_model_evaluation = y_5.iloc[model_evaluation_index]

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
cross_validation = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_5, y_5, groups = groups_5)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups_5[train_index]}")
    print(f"  Test:  index={test_index}, group={groups_5[test_index]}")

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

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_5, y_5, groups = groups_5)):
    X_model_selection = X_5.iloc[model_selection_index]
    X_model_evaluation = X_5.iloc[model_evaluation_index]
    y_model_selection = y_5.iloc[model_selection_index]
    y_model_evaluation = y_5.iloc[model_evaluation_index]

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
cross_validation = GroupShuffleSplit(n_splits = 5, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_5, y_5, groups = groups_5)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups_5[train_index]}")
    print(f"  Test:  index={test_index}, group={groups_5[test_index]}")

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

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_5, y_5, groups = groups_5)):
    X_model_selection = X_5.iloc[model_selection_index]
    X_model_evaluation = X_5.iloc[model_evaluation_index]
    y_model_selection = y_5.iloc[model_selection_index]
    y_model_evaluation = y_5.iloc[model_evaluation_index]

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
cross_validation = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_5, y_5, groups = groups_5)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups_5[train_index]}")
    print(f"  Test:  index={test_index}, group={groups_5[test_index]}")

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

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_5, y_5, groups = groups_5)):
    X_model_selection = X_5.iloc[model_selection_index]
    X_model_evaluation = X_5.iloc[model_evaluation_index]
    y_model_selection = y_5.iloc[model_selection_index]
    y_model_evaluation = y_5.iloc[model_evaluation_index]

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
cross_validation = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

"Print the cross validation data split for manual verification"
for i, (train_index, test_index) in enumerate(cross_validation.split(X_5, y_5, groups = groups_5)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups_5[train_index]}")
    print(f"  Test:  index={test_index}, group={groups_5[test_index]}")

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

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_5, y_5, groups = groups_5)):
    X_model_selection = X_5.iloc[model_selection_index]
    X_model_evaluation = X_5.iloc[model_evaluation_index]
    y_model_selection = y_5.iloc[model_selection_index]
    y_model_evaluation = y_5.iloc[model_evaluation_index]

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

# Define cross validation split characteristics
cross_validation = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cross_validation.get_n_splits()

# Print the cross validation data split for manual verification
for i, (train_index, test_index) in enumerate(cross_validation.split(X_5, y_5, groups = groups_5)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups_5[train_index]}")
    print(f"  Test:  index={test_index}, group={groups_5[test_index]}")

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

for i, (model_selection_index, model_evaluation_index) in enumerate (cross_validation.split(X_5, y_5, groups = groups_5)):
    X_model_selection = X_5.iloc[model_selection_index]
    X_model_evaluation = X_5.iloc[model_evaluation_index]
    y_model_selection = y_5.iloc[model_selection_index]
    y_model_evaluation = y_5.iloc[model_evaluation_index]

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
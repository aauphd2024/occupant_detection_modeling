# -*- coding: utf-8 -*-
"""
Kamilla Heimar Andersen
kahean@build.aau.dk

This script provides a basis for running a nested cross validation (NCV) procedure
Adjustment in the code will be needed to run the NCV
Packages are imported for flexible changes in the code

"""

##############################################################################
### PACKAGES ###
##############################################################################

import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
import seaborn as sns
import os
import time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
import random
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import matthews_corrcoef
from collections import defaultdict
from statistics import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss
import csv

##############################################################################
### LOAD DATA ###
##############################################################################

# Load the dataset from the CSV file
dataset = pd.read_csv('your location.csv')

# Convert the 'DateTime' column to datetime format
dataset['DateTime'] = pd.to_datetime(dataset['DateTime'], format='%d-%m-%y %H:%M')

#%%

# define your input and output (X and y) and groups for CV group split
# now its an empty list for demonstration

X = [] # for room type-based model input
y = [] # for room type-based model output
X_full_dataset = [] # for generalized model input
y_full_dataset = [] # for generalized model output
groups = []

##############################################################################
### CYCLIC ENCODING ###
##############################################################################

# define a function for sin transformer
def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# define a function for cos transformer
def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# for day number of the week
X["day_sin"] = sin_transformer(7).fit_transform(X)["Day number of the week"]
X["day_cos"] = cos_transformer(7).fit_transform(X)["Day number of the week"]

# for hour number of the day
X["hour_sin"] = sin_transformer(24).fit_transform(X)["Hour of the day"]
X["hour_cos"] = cos_transformer(24).fit_transform(X)["Hour of the day"]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
X[["day_sin", "day_cos"]].plot(ax=ax[0])
X[["hour_sin", "hour_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation");

#%%
##############################################################################
### DEFINE A RATIO FOR HYPER PARAMETERS ###
##############################################################################

# Calculate the ratio of negative to positive examples
ratio = np.sum(y == 0) / np.sum(y == 1)
print(ratio)

#%%
##############################################################################
### NESTED CROSS-VALIDATION (STEP 3) ###
##############################################################################

"Define the type of model to be trained"
xgb_model = xgb.XGBClassifier(objective='binary:logistic')  # XGBoost random forest model

"Define the hyperparameters you want to tune and their possible values"
# around 115.000 combinations
params_grid = {
    'learning_rate': np.linspace(0.01, 0.5, 4),
    'max_depth': [3, 5, 6, 8],
    'n_estimators': [50, 100, 150, 200],
    'gamma': np.linspace(0, 0.5, 3),
    'subsample': np.linspace(0.5, 1.0, 3),
    'colsample_bytree': np.linspace(0.5, 1.0, 3),
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': np.linspace(max(2.8845598845598848/2, 1), 2*2.8845598845598848, 5)
}
    
"Define outer loop split characteristics"
outer_loop = GroupShuffleSplit(n_splits=10, random_state=42, test_size = 0.2)
outer_loop.get_n_splits()

"Print outer loop data split for manual verification"
for i, (train_index, test_index) in enumerate(outer_loop.split(X, y, groups = groups)):
    print(f"Fold in outer loop {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

"Define inner loop (gridsearch with cross-validation) split characteristics"
inner_cv = GroupShuffleSplit(n_splits=10, random_state=42, test_size = 0.2)
inner_cv.get_n_splits()

"Print inner loop data split for manual verification"
for i, (train_index, test_index) in enumerate(inner_cv.split(X, y, groups = groups)):
    print(f"Fold in inner loop {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

#%% -> Start of the nested cross-validation (loop)

# Start the timer
start_time = time.time()

# Create an empty list to store the metrics for each fold
metrics = []

# Start optimization
train_accuracies = []
test_accuracies = []

# Initialize an empty DataFrame to store feature importances
feature_importances = pd.DataFrame(index=X.columns)

# Create a loop to make train test with group shuffle split
for i, (model_selection_index, model_evaluation_index) in enumerate(outer_loop.split(X, y, groups = groups)):
    
    # Get the current slit datasets for X and Y
    X_model_selection = X.iloc[model_selection_index].reset_index(drop=True)
    X_model_evaluation = X.iloc[model_evaluation_index].reset_index(drop=True)
    y_model_selection = y.iloc[model_selection_index].reset_index(drop=True)
    y_model_evaluation = y.iloc[model_evaluation_index].reset_index(drop=True)

    # here add the group split for the model_selection / evaluation - check the individual room code for that
    groups_model_selection = groups[model_selection_index]
    
    # Create the gridsearch (inner cross-validation for best hyper-parameter identification)
    grid_search = GridSearchCV(estimator=xgb_model, scoring='matthews_corrcoef', param_grid=params_grid, verbose = 3, cv=inner_cv, n_jobs=-1)

    # Fit the GridSearchCV (this executes the gridsearch to find the best hyper-parameters)
    grid_search.fit(X_model_selection, y_model_selection, groups = groups_model_selection)  # This the Inner loop !!!!!
    
    # Compute feature importances and store them
    fold_feature_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_model_selection.columns)
    feature_importances[f'fold_{i}'] = fold_feature_importances

    # Print the best model for this outer fold
    print(f"Best Model for outer fold {i}: ", grid_search.best_estimator_)  # Best estimator is the best model of the cross-validation GridSearh
    
    # The attribute best_params_ gives the best set of hyper-parameters that maximizes the mean score of the cross-validations
    print(f"The best parameters found for outer fold {i}: ", grid_search.best_params_)  # Hyper-parameters of the best model/estimator
    
    # the mean score obtained by using the parameters best_params_
    print(f"The mean CV score of the best model is: {grid_search.best_score_:.3f}")  # Best score is the average score of the cross-validation iteration of the best estimator found in the GridSearh

    # Predict and calculate the score of the best estimator/model on the model evaluation data
    y_pred = grid_search.best_estimator_.predict(X_model_evaluation)

    # Predict probabilities for the positive class
    y_prob = grid_search.best_estimator_.predict_proba(X_model_evaluation)[:, 1]
        
    # Generate and print the classification report...
    print(f"Classification Report for fold {i}:\n")
    print(classification_report(y_model_evaluation, y_pred))
    
    # Calculate and store the train error
    y_train_pred = grid_search.best_estimator_.predict(X_model_selection)
    train_accuracy = balanced_accuracy_score(y_model_selection, y_train_pred)
    train_accuracies.append(train_accuracy)

    # Predict and calculate the test error
    test_accuracy = balanced_accuracy_score(y_model_evaluation, y_pred)
    test_accuracies.append(test_accuracy)
    
    # Calculate score with balanced accuracy score
    score = balanced_accuracy_score(y_model_evaluation, y_pred)

    # Compute the metrics
    cm = confusion_matrix(y_model_evaluation, y_pred)
    precision = precision_score(y_model_evaluation, y_pred)
    recall = recall_score(y_model_evaluation, y_pred)
    f1score = f1_score(y_model_evaluation, y_pred)
    mcc = matthews_corrcoef(y_model_evaluation, y_pred)
    roc_auc = roc_auc_score(y_model_evaluation, y_pred)
    accuracy = accuracy_score(y_model_evaluation, y_pred)
    brier_loss = brier_score_loss(y_model_evaluation, y_prob)
    
    # Store the metrics in a dictionary
    metrics_dict = {}
    metrics_dict['best_params'] = grid_search.best_params_
    metrics_dict['score'] = score
    metrics_dict['cm'] = cm
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall
    metrics_dict['f1score'] = f1score
    metrics_dict['mcc'] = mcc
    metrics_dict['roc_auc'] = roc_auc
    metrics_dict['accuracy'] = accuracy
    metrics_dict['brier_loss'] = brier_loss
    
    # Append the dictionary to the list
    metrics.append(metrics_dict)

# Compute the mean feature importance across folds
feature_importances['mean'] = feature_importances.mean(axis=1)

# Print feature importances
print(feature_importances)

# Plot the train and test errors over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_accuracies)), train_accuracies, label='Train accuracy')
plt.plot(range(len(test_accuracies)), test_accuracies, label='Test accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
    
# Print the elapsed time
elapsed_time = time.time() - start_time
print("Elapsed Time for nested cross-validation (s):", elapsed_time)

#%%
##############################################################################
### METRIC CALCULATION ###
##############################################################################

# Print metrics for each fold
for i, metric in enumerate(metrics):
    print(f"Metrics for outer fold {i}:")
    print(f"  Best parameters: {metric['best_params']}")
    print(f"  Balanced accuracy score: {metric['score']}")
    print(f"  Confusion matrix:\n{metric['cm']}")
    print(f"  Precision: {metric['precision']}")
    print(f"  Recall: {metric['recall']}")
    print(f"  F1 score: {metric['f1score']}")
    print(f"  MCC: {metric['mcc']}")
    print(f"  AU_ROC: {metric['roc_auc']}")
    print(f"  Brier score loss: {metric['brier_loss']}")
    print(f"  Accuracy score: {metric['accuracy']}\n")

# Calculates average and std dev of all best models in each fold of the outer loop

# Initialize counters
balancedaccuracy_vect = []
precision_vect = []
recall_vect = []
f1_vect = []
mcc_vect = []
roc_auc_vect = []
accuracy_vect = []
brier_loss_vect = []
#confusion_matrix_vect = []

# Sum up all metrics
for metric in metrics:
    balancedaccuracy_vect.append(metric['score'])
    precision_vect.append(metric['precision'])
    recall_vect.append(metric['recall'])
    f1_vect.append(metric['f1score'])
    mcc_vect.append(metric['mcc'])
    roc_auc_vect.append(metric['roc_auc'])
    accuracy_vect.append(metric['accuracy'])
    brier_loss_vect.append(metric['brier_loss'])
    #confusion_matrix_vect.append(metric['cm'])
    
# Calculate averages
avg_balancedaccuracy = np.array(balancedaccuracy_vect).mean()
avg_precision = np.array(precision_vect).mean()
avg_recall = np.array(recall_vect).mean()
avg_f1 = np.array(f1_vect).mean()
avg_mcc = np.array(mcc_vect).mean()
avg_rocauc = np.array(roc_auc_vect).mean()
avg_accuracy = np.array(accuracy_vect).mean()
avg_brier_loss = np.array(brier_loss_vect).mean()
#avg_cm = np.array(confusion_matrix_vect).mean()

# Calculate std devs
std_dev_balancedaccuracy = np.std(np.array(balancedaccuracy_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_precision = np.std(np.array(precision_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_recall = np.std(np.array(recall_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_f1 = np.std(np.array(f1_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_mcc = np.std(np.array(mcc_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_rocauc = np.std(np.array(roc_auc_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_accuracy = np.std(np.array(accuracy_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population
std_dev_brierloss = np.std(np.array(brier_loss_vect), ddof=0) # Set ddof=0 to calculate the std dev of entire population and not a sample of population
#std_dev_cm = np.std(np.array(confusion_matrix_vect), ddof=0)  # Set ddof=0 to calculate the std dev of entire population and not a sample of population

# Initialize an empty confusion matrix
cumulative_cm = np.zeros_like(metrics[0]['cm'])

# Sum up all confusion matrices
for metric in metrics:
    cumulative_cm += metric['cm']

print(f"Cumulative confusion matrix:\n{cumulative_cm}")

# Print averages
print(f"Average balanced accuracy score: {avg_balancedaccuracy} with std dev of: {std_dev_balancedaccuracy}")
print(f"Average precision: {avg_precision} with std dev of: {std_dev_precision}")
print(f"Average recall: {avg_recall} with std dev of: {std_dev_recall}")
print(f"Average F1 score: {avg_f1} with std dev of: {std_dev_f1}")
print(f"Average MCC: {avg_mcc} with std dev of: {std_dev_mcc}")
print(f"Average ROC-AUC: {avg_rocauc} with std dev of: {std_dev_rocauc}")
print(f"Average accuracy: {avg_accuracy} with std dev of: {std_dev_accuracy}")
print(f"Average Brier loss: {avg_accuracy} with std dev of: {std_dev_brierloss}")

#%%
##############################################################################
### FINAL CROSS VALIDATION (STEP 4) ###
##############################################################################

# Define the cross-validator
final_cv = GroupShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42)

# Split the data
for i, (train_index, test_index) in enumerate(final_cv.split(X, y, groups = groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")
    
final_grid_search = GridSearchCV(estimator=xgb_model, scoring='matthews_corrcoef', param_grid=params_grid, verbose = 3, cv=final_cv, n_jobs=-1)

# Fit the GridSearchCV (this will fit the best model found in the inner loop)
final_grid_search.fit(X, y, groups = groups)  # Execute the GridSearch

# Print the best model for this outer fold
print("Best Model for final cross validation: ", final_grid_search.best_estimator_)

# The attribute best_params_ gives us the best set of parameters that maximize the mean score 
print("The best parameters found for final cross validation: ", final_grid_search.best_params_)


# Saves the best parameters in a .csv file for further use
# Define the name of the csv file
filename = "best_params.csv"

# Write the dictionary to the file
with open(filename, 'w') as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(["Parameter", "Value"])
    # Write the parameters
    for key, value in final_grid_search.best_params_.items():
        writer.writerow([key, value])

print(f"Best parameters have been saved to {filename}")
# the best parameters are to be used in the file "application_ODM.py"


# the mean score obtained by using the parameters best_params_
print("The mean CV score of the final cross validation model is: ", final_grid_search.best_score_)

# Get the best model
best_model = final_grid_search.best_estimator_

"Calculate the feature importance and plot figure"

# Calculate feature importances
importances = best_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()

##############################################################################
### SCRIPT END ###
##############################################################################
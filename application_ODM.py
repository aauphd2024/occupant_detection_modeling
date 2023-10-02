# -*- coding: utf-8 -*-

"""
KAMILLA HEIMAR ANDERSEN
kahean@build.aau.dk

This script applies the best parameters from the final cross validation 
Adjustment in the code will be needed to apply the models and get new labels out
Packages are imported for flexible changes in the code
The analysis is not included in the script
"""

#%%
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
from mrmr import mrmr_classif
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
import random
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from datetime import date, datetime, timedelta  # Date and time stamp generation
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import brier_score_loss
import matplotlib.dates as mdates

#%%
##############################################################################
### FOLDER LOCATION ###
##############################################################################

# Initialization
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

os.chdir(dname)
current_working_folder = os.getcwd()
print(current_working_folder)

#%%
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

X_full_dataset = []
y_full_dataset = []

#%%
##############################################################################
### DF PER ROOM ###
##############################################################################

df_room1 = dataset[dataset['room_type'] == 1]
df_room2 = dataset[dataset['room_type'] == 2]
df_room3 = dataset[dataset['room_type'] == 3]
df_room4 = dataset[dataset['room_type'] == 4]
df_room5 = dataset[dataset['room_type'] == 5]

#%%
##############################################################################
### BEDROOM ###
##############################################################################

X_1 = df_room1[['co2_concentration', 'air_temperature', 'relative_humidity', 'Hour of the day', 'Day number of the week', 'day_label', 'bedroom']]
y_1 = df_room1['occupancy_ground_truth']

# define a function for sin transformer
def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# define a function for cos transformer
def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# for day number of the week
X_1["day_sin"] = sin_transformer(7).fit_transform(X_1)["Day number of the week"]
X_1["day_cos"] = cos_transformer(7).fit_transform(X_1)["Day number of the week"]

# for hour number of the day
X_1["hour_sin"] = sin_transformer(24).fit_transform(X_1)["Hour of the day"]
X_1["hour_cos"] = cos_transformer(24).fit_transform(X_1)["Hour of the day"]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
X_1[["day_sin", "day_cos"]].plot(ax=ax[0])
X_1[["hour_sin", "hour_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation");

X_1.drop(['Hour of the day'], axis = 1, inplace = True)
X_1.drop(['Day number of the week'], axis = 1, inplace = True)
X_1.drop(['bedroom'], axis = 1, inplace = True)
X_1.drop(['day_label'], axis = 1, inplace = True)

#%%
##############################################################################
### OFFICE ###
##############################################################################

X_2 = df_room2[['co2_concentration', 'air_temperature', 'relative_humidity', 'Hour of the day', 'Day number of the week', 'day_label', 'office']]
y_2 = df_room2['occupancy_ground_truth']

# define a function for sin transformer
def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# define a function for cos transformer
def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# for day number of the week
X_2["day_sin"] = sin_transformer(7).fit_transform(X_2)["Day number of the week"]
X_2["day_cos"] = cos_transformer(7).fit_transform(X_2)["Day number of the week"]

# for hour number of the day
X_2["hour_sin"] = sin_transformer(24).fit_transform(X_2)["Hour of the day"]
X_2["hour_cos"] = cos_transformer(24).fit_transform(X_2)["Hour of the day"]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
X_2[["day_sin", "day_cos"]].plot(ax=ax[0])
X_2[["hour_sin", "hour_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation");

X_2.drop(['Hour of the day'], axis = 1, inplace = True)
X_2.drop(['Day number of the week'], axis = 1, inplace = True)
X_2.drop(['office'], axis = 1, inplace = True)
X_2.drop(['day_label'], axis = 1, inplace = True)

#%%
##############################################################################
### RUN MODEL WITH HYPER PARAMETERS (STEP 5) ###
##############################################################################
  
"GENERALIZED MODEL"

# group shuffle split
# {'colsample_bytree': 0.75, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 200, 'scale_pos_weight': 1.4422799422799424, 'subsample': 1.0}

# train the xgb model
final_model = XGBClassifier(scale_pos_weight = 1.4422799422799424, colsample_bytree=0.75, max_depth=6, learning_rate=0.01, gamma=0.5, min_child_weight=3, n_estimators=200, subsample=1)
final_model.fit(X_full_dataset, y_full_dataset)

#%%    

"BEDROOM MODEL"

# bedroom:
#    {'colsample_bytree': 0.75, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 8, 'min_child_weight': 5, 'n_estimators': 150, 'scale_pos_weight': 1.4422799422799424, 'subsample': 0.75}

# train the xgb model
final_model_bedroom = XGBClassifier(scale_pos_weight = 1.4422799422799424, colsample_bytree=0.75, max_depth=8, learning_rate=0.01, gamma=0, min_child_weight=5, n_estimators=150, subsample=0.75)
final_model_bedroom.fit(X_1, y_1)

#%%

"OFFICE MODEL"

# office: 
#   {'colsample_bytree': 1.0, 'gamma': 0.0, 'learning_rate': 0.33666666666666667, 'max_depth': 6, 'min_child_weight': 5, 'n_estimators': 50, 'scale_pos_weight': 2.5239898989898992, 'subsample': 1.0}

# train the xgb model
final_model_office = XGBClassifier(scale_pos_weight = 2.5239898989898992, colsample_bytree=1, max_depth=6, learning_rate=0.33666666666666667, gamma=0, min_child_weight=5, n_estimators=50, subsample=1)
final_model_office.fit(X_2, y_2)

#%%
##############################################################################
### INTRODUCE UNSEEN DATA ###
##############################################################################

# introducing new unseen data
new_data = pd.read_csv('data.csv')

new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], format='%d-%m-%y %H:%M')

#new_data['co2_concentration'] = new_data['co2_concentration'].interpolate()
#new_data['air_temperature'] = new_data['air_temperature'].interpolate()
#new_data['relative_humidity'] = new_data['relative_humidity'].interpolate()

#%%
##############################################################################
### ADD HOUR AND DAY ###
##############################################################################

"Hour and day data"
# make empty list for adding data later"
list_hour_of_the_day_data = []
list_day_nbr_of_the_week_data = []

for t in list(new_data['DateTime']):
    
    "Add hour of the day"
    hour_of_the_day = int(t.strftime("%H"))  # Convert the answer into integer
    list_hour_of_the_day_data.append(hour_of_the_day)
    
    "Add day number of the week"
    day_nbr_of_the_week = int(t.strftime("%w"))  # Day number of the week from 0 to 6 with Sunday as day 0
    if day_nbr_of_the_week == 0: day_nbr_of_the_week = 7  # Set Sunday with day number 7
    list_day_nbr_of_the_week_data.append(day_nbr_of_the_week)  # Day number of the week from 1 to 7
    
"Add new columns to the selected df"
new_data["Hour of the day"] = list_hour_of_the_day_data
new_data["Day number of the week"] = list_day_nbr_of_the_week_data

#%%

"some more data treatment to manually verify the dfs"

new_data_copy = new_data.copy()

new_data_copy.drop(['DateTime'], axis = 1, inplace = True)

occupancy = new_data_copy['occupancy'].values

#%%
##############################################################################
### CYCLIC ENCODING ON NEW DATA ###
##############################################################################

# define a function for sin transformer
def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# define a function for cos transformer
def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# for day number of the week
new_data_copy["day_sin"] = sin_transformer(7).fit_transform(new_data_copy)["Day number of the week"]
new_data_copy["day_cos"] = cos_transformer(7).fit_transform(new_data_copy)["Day number of the week"]

# for hour number of the day
new_data_copy["hour_sin"] = sin_transformer(24).fit_transform(new_data_copy)["Hour of the day"]
new_data_copy["hour_cos"] = cos_transformer(24).fit_transform(new_data_copy)["Hour of the day"]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16,8))
new_data[["day_sin", "day_cos"]].plot(ax=ax[0])
new_data[["hour_sin", "hour_cos"]].plot(ax=ax[1])
plt.suptitle("Cyclical encoding with sine/cosine transformation for new data");

#%%
##############################################################################
### REMOVE COLUMNS BASED ON GENERALIZED MODEL OR ROOM TYPE-BASED MODEL ###
##############################################################################

new_data_copy.drop(['Day number of the week'], axis = 1, inplace = True)
new_data_copy.drop(['Hour of the day'], axis = 1, inplace = True)
new_data_copy.drop(['occupancy'], axis = 1, inplace = True)
new_data_copy.drop(['room_no'], axis = 1, inplace = True)
new_data_copy.drop(['floor_area'], axis = 1, inplace = True)
#new_data_copy.drop(['apt_no'], axis = 1, inplace = True)
#new_data_copy.drop(['room_type'], axis = 1, inplace = True)

#new_data_copy.drop(['kitchen'], axis = 1, inplace = True)
#new_data_copy.drop(['livingroom'], axis = 1, inplace = True)
#new_data_copy.drop(['bedroom'], axis = 1, inplace = True)
#new_data_copy.drop(['office'], axis = 1, inplace = True)
#new_data_copy.drop(['kitchen_livingroom'], axis = 1, inplace = True)

print(new_data_copy.columns)

#%%
##############################################################################
### PREDICT NEW LABELS ON FULL NEW DATA ###
##############################################################################

"GENERALIZED MODEL"
predicted_labels = final_model.predict(new_data_copy)
# y_pred contains the predicted occupancy labels for the new data --> is predicted and stored as an array

#%%

"BEDROOM MODEL"
predicted_labels_bedroom = final_model_bedroom.predict(new_data_copy)
# y_pred contains the predicted occupancy labels for the new data --> is predicted and stored as an array

#%%

"OFFICE MODEL"
predicted_labels_office = final_model_office.predict(new_data_copy)
# y_pred contains the predicted occupancy labels for the new data --> is predicted and stored as an array

#%%
##############################################################################
### METRICS - COMPARISON IF GROUND TRUTH IS PRESENT ###
##############################################################################

# Compute the metrics without print in the computation itself
cm = confusion_matrix(occupancy, predicted_labels)
precision = precision_score(occupancy, predicted_labels)
recall = recall_score(occupancy, predicted_labels)
f1score = f1_score(occupancy, predicted_labels)
mcc = matthews_corrcoef(occupancy, predicted_labels)
roc_auc = roc_auc_score(occupancy, predicted_labels)
accuracy = accuracy_score(occupancy, predicted_labels)
balanced_accuracy = balanced_accuracy_score(occupancy, predicted_labels)
briar_score = brier_score_loss(occupancy, predicted_labels)

# Now print the results using f-strings
print(f"Confusion Matrix:\n{cm}")
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1score:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Brier Score Loss: {briar_score:.4f}")

##############################################################################
### SCRIPT END ###
##############################################################################
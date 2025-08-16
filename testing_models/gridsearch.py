### COMP9417 PROJECT ###
# Completed 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# Grid search algorithm to assist with finding best combinations of hyperparameters

### IMPORTS ###

from sklearn.model_selection import GridSearchCV, KFold

import pandas as pd

import model_rf

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
rf, y_val, y_pred = model_rf.model(X_train, y_train)

kf = KFold(n_splits=3, shuffle=False)

params = {
    'n_estimators'      : [200, 225, 250, 275, 300],
    'max_depth'         : [10, 15, 20, 25],
    'min_samples_split' : [6, 7, 8, 9],
    'min_samples_leaf'  : [1, 2, 3, 4]
}

grid_rf = GridSearchCV(rf, param_grid=params, cv=kf, 
                       scoring='f1_macro').fit(X_train, y_train)

print("Best parameters:", grid_rf.best_params_)
print("Best score:", grid_rf.best_score_)
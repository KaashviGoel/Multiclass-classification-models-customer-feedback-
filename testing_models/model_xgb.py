### COMP9417 PROJECT ###
# Completed 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# This code writes a logistic regression model from provided data

### IMPORTS ###

from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def model(
        X,
        y,
        feature_selection=0,
        smote_k_neighbors=2,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2
):
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)
    
    # apply feature selection
    if feature_selection:
        if feature_selection == 0:
            selector_model = XGBClassifier(objective='multi:softprob', num_class=28,
                                         eval_metric='mlogloss', random_state=42)
            selector = SelectFromModel(selector_model)
        else:
            # focus on f-score
            selector = SelectKBest(score_func=f_classif, k=feature_selection)

        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)

    # fit scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)

    # apply SMOTE with Tomek links if class size large enough
    class_counts = y_train.value_counts()
    min_class_size = class_counts.min()

    if min_class_size > 2:
        smote = SMOTETomek(smote=SMOTE(k_neighbors=smote_k_neighbors), random_state=200)
        X_train, y_train, smote.fit_resample(X_train, y_train)

    # train model
    xgb = XGBClassifier(
        objective='multi:softprob', 
        num_class=28, 
        eval_metric='mlogloss', 
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        random_state=42
    )

    # predictions
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_val)

    return xgb, y_val, y_pred
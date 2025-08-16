### COMP9417 PROJECT ###
# Completed 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# This code writes a logistic regression model from provided data

### IMPORTS ###

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

def model(
        X,
        y,
        feature_selection=0,
        smote_k_neighbors=2,
        num_class=28,
        boosting_type='gbdt',
        learning_rate=0.05,
        n_estimators=200,
        max_depth=8,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
):
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)
    


    # apply feature selection
    if feature_selection:
        if feature_selection == 0:
            selector_model = LGBMClassifier(class_weight='balanced', random_state=42)
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
        smote = SMOTETomek(smote=SMOTE(k_neighbors=2), random_state=200)
        X_train, y_train, smote.fit_resample(X_train, y_train)

    # train model
    lgb = LGBMClassifier(
        max_iter=2000,
        num_class=28,
        boosting_type='gbdt',
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42, 
        verbosity=-1
    )

    # predictions
    lgb.fit(X_train, y_train)
    y_pred = lgb.predict(X_val)

    return lgb, y_val, y_pred
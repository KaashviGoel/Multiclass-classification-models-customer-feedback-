### COMP9417 PROJECT ###
# Completed 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# This code writes a random forest model from provided data

from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

def model(
        X,
        y,
        feature_selection=0,
        smote_k_neighbors=2,
        n_estimators=250,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=4
):
    # split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)
    
    # apply feature selection
    if feature_selection:
        if feature_selection == 0:
            selector_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
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
    rf = RandomForestClassifier(
        n_estimators=n_estimators,              # num trees
        max_depth=max_depth,                    # tree size
        min_samples_split=min_samples_split,    # more splitting
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced_subsample', 
        random_state=42,
        n_jobs=-1
    )

    # predictions
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    return rf, y_val, y_pred



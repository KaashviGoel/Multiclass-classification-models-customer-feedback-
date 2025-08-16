Developed an ML pipeline to classify customer feedback into 28 departments using 300-dim NLP embeddings. Tackled class imbalance and distribution shifts with Logistic Regression, Random Forest, XGBoost, LightGBM, and MLP in a soft voting ensemble, applying class weighting, SMOTETomek, focal loss, and drift detection.
To run our final model, please ensure all .csv files are inside ensemble_models directory, then run ensemble.py for our original model, and ensemble_retrained.py for the model built for test 2. 

The repo contains 3 folders:

testing_models: holds the individual models we tested and tuned to get the best f1-scores we could, and eda materials.

ensemble_models: holds the files needed for our final model

ensemble_predictions: .npy files produced by the ensemble model

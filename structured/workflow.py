# How to determine model:
# Structured data = Ensemble methods
# Unstructured data = Deep learning or transfer learning
# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pickle

# 1) Get the data ready
heart_disease = pd.read_csv("data/heart-disease.csv")
# heart_disease_shuffled = heart_disease.sample(frac=1)

# Create X (features)
x = heart_disease.drop("target", axis=1)
# Create Y (labels)
y = heart_disease["target"]

# 2) Choose the right model and hyperparameters
clf = RandomForestClassifier()

# Set up RandomizedSearchCV
params = {
    "n_estimators": [10, 100, 200, 500],
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ["auto", "sqrt"],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
}
rs_clf = GridSearchCV(estimator=clf, param_grid=params, cv=5, verbose=2)

# View optimized hyperparameters
# rs_clf.best_params_

# 3) Fit the model to the data to train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
rs_clf.fit(x_train, y_train)

# Make a prediction
y_preds = rs_clf.predict(x_test)

# 4) Evaluate the model (Cross validation)
cvs = cross_val_score(rs_clf, x, y, scoring=None)
np.mean(cvs)

# %%

# 5) Improve a model
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} n_estimators")
    clf = RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) * 100:.2f}%")
    print("")

# 6) Save a model and load it
pickle.dump(clf, open("random_forst_model.pk1", "wb"))
loaded_model = pickle.load(open("random_forst_model.pk1", "rb"))

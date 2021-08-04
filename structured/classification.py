# Classification models
# 1) Logistic Regression
# 2) K-Nearest neighbours classifier
# 3) Random Forest classifier

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Import dataset
heart_disease = pd.read_csv("data/heart-disease.csv")

# Setup random seed
np.random.seed(42)

# Split into features and labels
x = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Build model
model = LinearSVC(max_iter=1000)
model.fit(x_train, y_train)
y_preds = model.predict(x_test)

# ROC Curve
y_probs = model._predict_proba_lr(x_test)
y_probs_positive = y_probs[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
roc_auc_score(y_test, y_probs_positive)

# Confusion matrix
confusion_matrix(y_test, y_preds)
pd.crosstab(y_test, y_preds, rownames=["Actual labels"], colnames=["Predicated Labels"])

# Classification report
print(classification_report(y_test, y_preds))

# %%

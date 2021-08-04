# Evaluation metrics:
# 1) Accurary
# 2) Area under ROC curve
# 3) Confusion matric
# 4) Classification matrix

#%%

# Data analysis and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluations
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    plot_roc_curve,
)

# Load data
df = pd.read_csv("data/heart-disease.csv")
# Check for missing values
# df.isna().sum()
# Overall desciption
# df.describe()
# Correlation matrix
# df.corr()

# Split data into X and Y
x = df.drop("target", axis=1)
y = df["target"]
# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Store different types of models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
}

# Train and evaluate best model from accuracy score
def fit_and_score():
    # Keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Train model
        model.fit(x_train, y_train)
        # Evaluate model
        model_scores[name] = model.score(x_test, y_test)
    return model_scores


# Plot a confusion matrix
def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")


# Create a cross-validated evaluation report
def cross_evaluation_report(model, numOfCV):
    cv_acc = cross_val_score(model, x, y, cv=numOfCV, scoring="accuracy")
    cv_precision = cross_val_score(model, x, y, cv=numOfCV, scoring="precision")
    cv_recall = cross_val_score(model, x, y, cv=numOfCV, scoring="recall")
    cv_f1 = cross_val_score(model, x, y, cv=numOfCV, scoring="f1")
    return pd.DataFrame(
        {
            "Accuracy": np.mean(cv_acc),
            "Precision": np.mean(cv_precision),
            "Recall": np.mean(cv_recall),
            "F1": np.mean(cv_f1),
        },
        index=[0],
    )


# Create a correlation chart
def correlation_chart():
    feature_dict = dict(zip(df.columns, list(model.coef_[0])))
    # Visualize feature importance via graph
    feature_df = pd.DataFrame(feature_dict, index=[0])
    feature_df.T.plot.bar(title="Feature importance", legend=False)


# Train model
model = models["Logistic Regression"]
model.fit(x_test, y_test)
# Predict
y_preds = model.predict(x_test)

# ROC curve
# plot_roc_curve(model, x_test, y_test)

# Confusion matrix
# plot_confusion_matrix()

# Classification report
# cross_evaluation_report(model, 5)

# Correlation among columns
correlation_chart()

# %%

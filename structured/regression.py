# Evaluation metrics:
# 1) Coefficient of determiniation (R^2)
# 2) Mean absolute error (MAE)
# 3) Mean squared error (MSE)

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import dataset
boston = load_boston()

# Convert to panda dataframe
boston_df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
boston_df["target"] = pd.Series(boston["target"])

# Setup random seed
np.random.seed(42)

# Split into features and labels
x = boston_df.drop("target", axis=1)
y = boston_df["target"]

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Build model
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_preds = model.predict(x_test)

# R^2
model.score(x_test, y_test)

# MAE - Average of the absolute diff between predictions and actual values
mae = mean_absolute_error(y_test, y_preds)
df = pd.DataFrame(data={"Actual values": y_test, "Predicted values": y_preds})
df["differences"] = df["Predicted values"] - df["Actual values"]

# MSE
mse = mean_squared_error(y_test, y_preds)
mse

# %%

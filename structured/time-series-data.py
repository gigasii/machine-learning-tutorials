import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Import data (Parse dates)
df = pd.read_csv(
    "data2/TrainAndValid.csv",
    low_memory=True,
    parse_dates=["saledate"],
)

# Sort Dataframe by date
df.sort_values(by=["saledate"], ascending=True, inplace=True)

# Make a copy of original dataframe
df_temp = df.copy()

# Feature engineering
def preprocess_data(df):
    # Split date into separate columns
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    # Drop redundant date column
    df.drop("saledate", axis=1, inplace=True)

    # Convert Object (Strings) columns into pandas' category
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label] = content.astype("category").cat.as_ordered()

    # Fill missing data
    for label, content in df.items():
        # Numeric columns
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing
                df[f"{label}_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
        # Categorical columns
        elif pd.api.types.is_categorical_dtype(content):
            # Add binary column to indicate whether sample had missing value
            df[f"{label}_is_missing"] = pd.isnull(content)
            # Turn categories into numbers and add +1
            df[label] = pd.Categorical(content).codes + 1

    return df


df_temp = preprocess_data(df_temp)

# Split data into train and validation
df_valid = df_temp[df_temp.saleYear == 2012]
df_train = df_temp[df_temp.saleYear != 2012]
# Create labels
x_valid, y_valid = df_valid.drop("SalePrice", axis=1), df_valid["SalePrice"]
x_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]

# Calculates root mean squared log error between predictions and true labels
def rmsle(labels, preds):
    return np.sqrt(mean_squared_log_error(labels, preds))


# Evaluate model on a few different levels
def show_scores(model):
    train_preds = model.predict(x_train)
    val_preds = model.predict(x_valid)
    scores = {
        "Training MAE": mean_absolute_error(y_train, train_preds),
        "Valid MAE": mean_absolute_error(y_valid, val_preds),
        "Training RMSLE": rmsle(y_train, train_preds),
        "Valid RMSLE": rmsle(y_valid, val_preds),
        "Training R^2": r2_score(y_train, train_preds),
        "Valid R^2": r2_score(y_valid, val_preds),
    }
    return scores


# Create model
model = RandomForestRegressor(n_jobs=-1, random_state=42, max_samples=10000)

# Cutting down max number of samples each estimator can see improves training time
model.fit(x_train, y_train)

# Hyperparameter tuning
rf_grid = {
    "n_estimators": np.arange(10, 100, 10),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
    "max_features": [0.5, 1, "sqrt", "auto"],
    "max_samples": [10000],
}
rs_model = RandomizedSearchCV(
    model, param_distributions=rf_grid, n_iter=2, cv=5, verbose=True
)

# Train model
rs_model.fit(x_train, y_train)

# Import test data
df_test = pd.read_csv("data2/Test.csv", low_memory=False, parse_dates=["saledate"])
df_test = preprocess_data(df_test)

# Determine the column difference
set(x_train.columns) - set(df_test.columns)
df_test["auctioneerID_is_missing"] = False

# Predictions
test_preds = rs_model.predict(df_test)

# Plot feature importance
def plot_features(columns, importances, n=20):
    df = (
        pd.DataFrame({"features": columns, "feature_importances": importances})
        .sort_values("feature_importances", ascending=False)
        .reset_index(drop=True)
    )
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Features")


#%%

plot_features(x_train.columns, rs_model.best_estimator_.feature_importances_)


#%%

#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Import data
car_sales = pd.read_csv("data/car-sales-extended-missing-data.csv")

# Clean data (Drop rows with missing labels)
car_sales.dropna(subset=["Price"], inplace=True)

# Define imputers
categorical_inputer = SimpleImputer(strategy="constant", fill_value="missing")
door_inputer = SimpleImputer(strategy="constant", fill_value=4)
num_inputer = SimpleImputer(strategy="mean")
# Define columns
categorical_features = ["Make", "Colour"]
door_features = ["Doors"]
num_features = ["Odometer (KM)"]
# Define different transformer pipeline
categorical_transformer = Pipeline(
    steps=[
        ("imputer", categorical_inputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
door_transformer = Pipeline(steps=[("imputer", door_inputer)])
num_transformer = Pipeline(steps=[("imputer", num_inputer)])

# Setup preprocessing steps (Fill missing values, then convert to numeric)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("door", door_transformer, door_features),
        ("num", num_transformer, num_features),
    ]
)

# Create a preprocessing and modelling pipeline
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor())]
)

# Split into features and labels
x = car_sales.drop("Price", axis=1)
y = car_sales["Price"]

# Split into training and test
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Tune hyperparameters
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "model__n_estimators": [100, 1000],
    "model__max_depth": [None, 5],
    "model__max_features": ["auto"],
    "model__min_samples_split": [2, 4],
}
optimized_model = GridSearchCV(model, param_grid, cv=5, verbose=2)

# Train model
optimized_model.fit(x_train, y_train)
optimized_model.score(x_test, y_test)


# %%

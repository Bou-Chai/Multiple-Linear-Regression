import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv("./data/winequality-red.csv", sep=";")

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

# Normalize features
min_max_scaler = preprocessing.MinMaxScaler()
feature_array = min_max_scaler.fit_transform(features)

#feature_array = features.values
target_array = target.values

# Split data for training and testing
feature_train, feature_test, target_train, target_test = train_test_split(feature_array, target_array, train_size=0.6)

# Train model
mlr_model = LinearRegression()
mlr_model.fit(feature_train, target_train)

# Get model predications on testing data
model_predictions = mlr_model.predict(feature_test)

# Get zero-rule predictions
target_train_mean = np.array(target_train).mean()
zero_rule_predictions = [target_train_mean for i in range(len(model_predictions))]

# Zero-rule metrics
mae_zero_rule = mean_absolute_error(target_test, zero_rule_predictions)
rmse_zero_rule = root_mean_squared_error(target_test, zero_rule_predictions)

# Model metrics
mae_model = mean_absolute_error(target_test, model_predictions)
rmse_model = root_mean_squared_error(target_test, model_predictions)

# Print metrics
print(f"MAE of zero-rule: {mae_zero_rule}")
print(f"RMSE of zero-rule: {rmse_zero_rule}")
print(f"MAE of model: {mae_model}")
print(f"RMSE of model: {rmse_model}")

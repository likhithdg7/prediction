import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the data
train_data = pd.read_csv(r'C:\Users\likit\Desktop\1\Box-Office-Revenue-Prediction-Using-Linear-Regression-in-ML\input\train.csv')
test_data = pd.read_csv(r'C:\Users\likit\Desktop\1\Box-Office-Revenue-Prediction-Using-Linear-Regression-in-ML\input\test.csv')

# Select relevant columns for prediction
columns_to_use = ['budget', 'popularity', 'runtime']

def preprocess_data(data):
    """Preprocess the data: handle missing values and extract relevant columns."""
    data = data[columns_to_use + ['revenue'] if 'revenue' in data.columns else columns_to_use]
    data = data.replace(0, np.nan)  # Replace 0s with NaN for better handling
    data = data.dropna()  # Drop rows with missing values
    return data

# Preprocess train and test datasets
train_data = preprocess_data(train_data)
X = train_data[columns_to_use]
y = train_data['revenue']

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
print(f"Mean Squared Error on validation set: {mse}")

# Save the model to a .pkl file
with open('box_office_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model has been trained and saved as 'box_office_model.pkl'")

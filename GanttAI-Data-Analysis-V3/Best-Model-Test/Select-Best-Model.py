import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Read the data
data = pd.read_csv(os.path.join(__location__, "dataset.csv"))

# Select features as input and duration as the model output
X = data[['size', 'complexity', 'price']]
y = data['duration']

# Convert categorical variables to binary variables
X = pd.get_dummies(X)

# Split the data into training (50 records) and testing (16 records) sets
X_train, X_test, y_train, y_test = X[:50], X[50:], y[:50], y[50:]

# Define a list of regressors
regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Support Vector Regressor": SVR(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "XGBoost Regressor": XGBRegressor(objective='reg:squarederror')
}

best_model = None
best_avg_r2 = float('-inf')

# Train and evaluate each regressor using KFold cross validation
for name, regressor in regressors.items():
    kf = KFold(n_splits=5)
    r2_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        regressor.fit(X_train_fold, y_train_fold)  # Train the regressor
        y_pred = regressor.predict(X_val_fold)  # Predict on validation data
        r2 = r2_score(y_val_fold, y_pred)  # Calculate R2 score
        r2_scores.append(r2)
    avg_r2 = np.mean(r2_scores)
    print(f"{name} Average R2 Score on validation set:", avg_r2)
    if avg_r2 > best_avg_r2:
        best_avg_r2 = avg_r2
        best_model = regressor

print("Best Model:", best_model)
print("Best Average R2 Score:", best_avg_r2)


# Define the neural network architecture

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Select features as input and duration as the model output
X = data[['size', 'complexity', 'price']]
y = data['duration']

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

# Define the dataset and data loader
dataset = TensorDataset(X_tensor, y_tensor)

# Define a list of models
models = {
    "Neural Network": NeuralNetwork(X.shape[1])
}

best_model = None
best_avg_r2 = float('-inf')

# Train and evaluate each model using KFold cross validation
for name, model in models.items():
    kf = KFold(n_splits=5)
    r2_scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X_tensor[train_index], X_tensor[val_index]
        y_train_fold, y_val_fold = y_tensor[train_index], y_tensor[val_index]

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(1000):  # You can adjust the number of epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_fold)
            loss = criterion(outputs, y_train_fold)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_fold)
            r2 = r2_score(y_val_fold.numpy(), y_pred.numpy())
            r2_scores.append(r2)

    avg_r2 = np.mean(r2_scores)
    print(f"{name} Average R2 Score on validation set:", avg_r2)
    if avg_r2 > best_avg_r2:
        best_avg_r2 = avg_r2
        best_model = model

print("Best Model:", best_model)
print("Best Average R2 Score:", best_avg_r2)


# Create a figure and axis object
fig, ax = plt.subplots()

# Define lists to store the names and average R2 scores
names = []
avg_r2_scores = []

# Train and evaluate each regressor using KFold cross validation
for name, regressor in regressors.items():
    kf = KFold(n_splits=5)
    r2_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        regressor.fit(X_train_fold, y_train_fold)  # Train the regressor
        y_pred = regressor.predict(X_val_fold)  # Predict on validation data
        r2 = r2_score(y_val_fold, y_pred)  # Calculate R2 score
        r2_scores.append(r2)
    avg_r2 = np.mean(r2_scores)
    names.append(name)
    avg_r2_scores.append(avg_r2)

# Plot the average R2 scores with distinct colors
bars = ax.bar(names, avg_r2_scores, color=['blue' if score == max(avg_r2_scores) else 'skyblue' for score in avg_r2_scores])

# Add labels for each bar
for bar, score in zip(bars, avg_r2_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.03, f'{score:.4f}', ha='center', color='white' if score == max(avg_r2_scores) else 'black')

# Set axis labels and title
ax.set_xlabel('Regressor')
ax.set_ylabel('Average R2 Score')
ax.set_title('Average R2 Scores for Different Regressors')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


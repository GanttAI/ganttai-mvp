import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

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

# Ensure all features are present in both datasets
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict durations for the test data
y_pred = model.predict(X_test)

# Predict durations for the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the mean squared error for both sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error (Training Data):", mse_train)
print("Mean Squared Error (Testing Data):", mse_test)

# Calculate the coefficient of determination (R^2 score) for both sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print("Coefficient of Determination (R^2 Score) (Training Data):", r2_train)
print("Coefficient of Determination (R^2 Score) (Testing Data):", r2_test)

# Predict durations for a set of new inputs
inputs = pd.DataFrame([
    {'location': 'Peninsula', 'size': 5000, 'complexity': 'high', 'price': 1000000.0},
    {'location': 'Marsh Island', 'size': 5529, 'complexity': 'high', 'price': 1221345.0}
])

# Convert categorical variables to binary variables
inputs = pd.get_dummies(inputs)

# Ensure all features are present in the new inputs
inputs = inputs.reindex(columns=X_train.columns, fill_value=0)

# Predict durations for the new inputs
predicted_duration = model.predict(inputs)

# Print the predicted durations
print("Predicted Durations:", predicted_duration)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Scatter plot for training data
axs[0].scatter(y_train, y_train_pred, color='blue', label='Actual vs Predicted (Training)')
axs[0].set_xlabel('Actual Duration')
axs[0].set_ylabel('Predicted Duration')
axs[0].set_title('Actual vs Predicted Durations (Training Data)', weight='bold')
axs[0].legend()

# Scatter plot for testing data
axs[1].scatter(y_test, y_test_pred, color='red', label='Actual vs Predicted (Testing)')
axs[1].set_xlabel('Actual Duration')
axs[1].set_ylabel('Predicted Duration')
axs[1].set_title('Actual vs Predicted Durations (Testing Data)', weight='bold')
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Calculate residuals for training data
residuals_train = y_train - y_train_pred

# Calculate absolute standardized residuals for training data
abs_std_residuals_train = np.abs((residuals_train - np.mean(residuals_train)) / np.std(residuals_train))

# Identify outliers for training data
outliers_train = abs_std_residuals_train > 3

# Calculate residuals for test data
residuals_test = y_test - y_test_pred

# Calculate absolute standardized residuals for test data
abs_std_residuals_test = np.abs((residuals_test - np.mean(residuals_train)) / np.std(residuals_train))

# Identify outliers for test data
outliers_test = abs_std_residuals_test > 3

# Plot scatter plot for training data with regression line and identify outliers
plt.figure(figsize=(15, 5))

# Plot for training data
plt.subplot(1, 2, 1)

# Plot scatter plot for training data
plt.scatter(y_train, y_train_pred, color='blue', label=f'Training Data (n={len(y_train)})')

# Plot regression line
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Regression Line')

# Annotate plot with R2 score
plt.annotate(f'[R2: {r2_train:.2f}]', xy=(0.40, 0.95), xycoords='axes fraction', fontsize=12)

# Plot outliers
plt.scatter(y_train[outliers_train], y_train_pred[outliers_train], color='red', label='Outliers')

plt.xlabel('Actual Durations')
plt.ylabel('Predicted Durations')
plt.title('Training Data: Actual vs. Predicted Durations', weight='bold')
plt.legend()
plt.grid(True)

# Plot for test data
plt.subplot(1, 2, 2)

# Plot scatter plot for test data
plt.scatter(y_test, y_test_pred, color='green', label=f'Test Data (n={len(y_test)})')

# Plot regression line
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Regression Line')

# Annotate plot with R2 score
plt.annotate(f'[R2: {r2_train:.2f}]', xy=(0.40, 0.95), xycoords='axes fraction', fontsize=12)

# Plot outliers
plt.scatter(y_test[outliers_test], y_test_pred[outliers_test], color='red', label='Outliers')

plt.xlabel('Actual Durations')
plt.ylabel('Predicted Durations')
plt.title('Test Data: Actual vs. Predicted Durations', weight='bold')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Predict durations for all data
y_all_pred = model.predict(X)

# Scatter plot for all data
plt.figure(figsize=(10, 8))
plt.scatter(y, y_all_pred, color='blue')
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.title('Actual vs Predicted Durations (All Data)', weight='bold')

# Plot regression line
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Regression Line')

# Annotate plot with R2 score
plt.annotate(f'[R2: {r2_score(y, y_all_pred):.2f}]', xy=(0.40, 0.95), xycoords='axes fraction', fontsize=12)

plt.legend()
plt.grid(True)
plt.show()

# Load the trained model
model = joblib.load(os.path.join(__location__, "model-01.pkl"))

# Read the data
data = pd.read_csv(os.path.join(__location__, "dataset.csv"))

# Select features as input and duration as the model output
X = data[['size', 'complexity', 'price']]
y = data['duration']

# Convert categorical variables to binary variables
X = pd.get_dummies(X)

# Predict durations for all data
y_pred = model.predict(X)

# Calculate R2 score
r2 = r2_score(y, y_pred)

# Calculate tolerance
tolerance = np.sqrt((3 - r2) * np.var(y_pred))  # You can adjust this value based on your requirements

# Scatter plot for all data
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred, color='blue')
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.title(f'Actual vs Predicted Durations (All Data for Trained Model)\nR2 Score: {r2:.2f}',weight='bold')

# Plot regression line
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Regression Line')

# Calculate upper and lower lines with tolerance
upper_line = [min(y) + tolerance, max(y) + tolerance]
lower_line = [min(y) - tolerance, max(y) - tolerance]

# Plot upper and lower lines
plt.plot([min(y), max(y)], upper_line, color='gray', linestyle='--', label='Upper Line (Tolerance)')
plt.plot([min(y), max(y)], lower_line, color='gray', linestyle='--', label='Lower Line (Tolerance)')

# Find outliers
outliers = np.abs(y - y_pred) > tolerance
plt.scatter(y[outliers], y_pred[outliers], color='red', label='Outliers')

# Annotate plot with R2 score
plt.annotate(f'[R2: {r2:.2f}]', xy=(0.40, 0.90), xycoords='axes fraction', fontsize=12)

plt.legend()
plt.grid(True)
plt.show()

# Calculate the R^2 score for training and testing data
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the R^2 scores
print("R^2 Score (Training Data):", r2_train)
print("R^2 Score (Testing Data):", r2_test)

# Optionally, you can convert the R^2 scores to percentages
r2_train_percentage = r2_train * 100
r2_test_percentage = r2_test * 100

print("R^2 Score (Training Data): {:.2f}%".format(r2_train_percentage))
print("R^2 Score (Testing Data): {:.2f}%".format(r2_test_percentage))

# Calculate R2 score
r2 = r2_score(y, y_pred)

# Calculate MSE
mse = mean_squared_error(y, y_pred)

print("R-squared (R^2) Score:", r2)
print("Mean Squared Error (MSE):", mse)

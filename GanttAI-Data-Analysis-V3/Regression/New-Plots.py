import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Read the data
data = pd.read_csv(os.path.join(__location__, "dataset.csv"))

# Select features as input and duration as the model output
X = data[['size', 'complexity', 'price', 'location']]
y = data['duration']


# Convert categorical variables to binary variables
X = pd.get_dummies(X)

# Split the data into training (50 records) and testing (16 records) sets
X_train, X_test, y_train, y_test = X[:49], X[49:], y[:49], y[49:]

# Ensure all features are present in both datasets
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# Create a linear regression model
model = Lasso()

# Train the model
model.fit(X_train, y_train)

# Predict durations for the test data
y_pred = model.predict(X_test)

features = ['price', 'size', 'complexity']

# Define a function to plot regression line along with scatter plot


def plot_regression(x, y, title):
    plt.scatter(x, y, color='blue')
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Duration')
    plt.grid(True)


# Plot actual durations with each feature
plt.figure(figsize=(16, 6))

for i, feature in enumerate(features):
    plt.subplot(2, 4, i+1)
    plt.scatter(X_test[feature], y_test, color='blue')
    plt.title('Actual Duration vs {}'.format(feature))
    plt.xlabel(feature)
    plt.ylabel('Duration')
    plt.grid(True)

# Plot predicted durations with each feature
for i, feature in enumerate(features):
    plt.subplot(2, 4, i+5)
    plt.scatter(X_test[feature], y_pred, color='blue')
    plt.title('Predicted Duration vs {}'.format(feature))
    plt.xlabel(feature)
    plt.ylabel('Duration')
    plt.grid(True)

plt.tight_layout()
plt.show()

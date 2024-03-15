import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os,pandas as pd

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
X_train, X_test, y_train, y_test = X[:49], X[49:], y[:49], y[49:]

# Initialize the KNN regressor with a chosen number of neighbors
knn_regressor = KNeighborsRegressor(n_neighbors=15)

# Train the KNN regressor on the training data
knn_regressor.fit(X_train, y_train)

# Predict the duration for the test set
y_pred = knn_regressor.predict(X_test)

# Calculate mean squared error as a performance metric
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R2 score as a performance metric
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Plot actual and predicted durations
plt.scatter(y_test, y_pred, color='blue', label='Predicted')
plt.scatter(y_test, y_test, color='red', label='Actual')
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.title('Actual vs Predicted Durations (R2 Score: {:.2f})'.format(r2))
plt.legend()

# Add R2 score as text on the plot
plt.text(100, 800, 'R2 = {:.2f}'.format(r2), fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

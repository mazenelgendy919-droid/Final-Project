# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. Load the dataset
data = pd.read_csv('student_scores.csv')
print("First 5 rows of the dataset:")
print(data.head())
print("\nMissing values in each column:")
print(data.isnull().sum())


# 2. Preprocessing
# Remove rows with missing values (if any)
data = data.dropna()


# 3. Visualization (Bar Chart)
plt.figure()
plt.bar(data['Hours_Studied'], data['Final_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Final Score')
plt.title('Hours Studied vs Final Score')
plt.show()


# 4. Select features and target
X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Final_Score']


# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=29
)


# 6. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# 7. Evaluate the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2:.2f}%")


# 8. Display predictions vs actual values (for verification)
print("\nSample Predictions vs Actual Values:")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.2f} | Actual: {y_test.values[i]}")


# Keep the terminal open
input("\nPress Enter to exit...")

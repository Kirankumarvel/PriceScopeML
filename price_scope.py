import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

#Load the California Housing data set
# Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target
#Print the description of the California Housing data set
print(data.DESCR)

#task 1. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Explore the training data
eda = pd.DataFrame(data=X_train)
eda.columns = data.feature_names
eda['MedHouseVal'] = y_train
print(eda.describe())

# Check for missing values
print("Missing values in training data:\n", eda.isnull().sum())

# Skewness of each column
print("\nSkewness:")
print(eda.skew())

#Task 2: What range are most of the median house prices valued at

q1 = eda['MedHouseVal'].quantile(0.25)
q3 = eda['MedHouseVal'].quantile(0.75)

print(f"Most median house prices fall between ${q1 * 100000:.0f} and ${q3 * 100000:.0f}")



#How are the median house prices distributed?

# Plot the distribution
plt.hist(1e5*y_train, bins=30, color='lightblue', edgecolor='black')
plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

#Model fitting and prediction

# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_regressor.predict(X_test)
# Removed redundant re-initialization of RandomForestRegressor
#Estimate out-of-sample MAE, MSE, RMSE, and R²
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Summary Output
print(f"Mean Absolute Error (MAE): ${mae * 1e5:.0f}")
print(f"Root Mean Squared Error (RMSE): ${rmse * 1e5:.0f}")
print(f"R² Score: {r2:.2f}")
print("\nSummary:")
print("On average, predictions are off by about ${:,.0f}.".format(mae * 1e5))
print("Larger errors go up to around ${:,.0f}.".format(rmse * 1e5))
print(f"The model explains {r2 * 100:.0f}% of the variance in house prices.")
print("However, deeper insights are needed to know where it performs well or poorly before reporting to stakeholders.")

#Plot Actual vs Predicted values
# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Task 2. Plot the histogram of the residual errors (dollars)
residuals = y_test - y_pred_test

# Plot histogram of residuals (in dollars)
# Calculate residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals * 1e5, bins=30, color='salmon', edgecolor='black')
plt.title("Histogram of Residual Errors (in $)")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print mean and standard deviation of residuals
mean_resid = np.mean(residuals)
std_resid = np.std(residuals)

print(f"Mean of residuals: {mean_resid:.4f}")
print(f"Standard deviation of residuals: {std_resid:.4f}")

#Task 3: Plot the model residual errors by median house value.
# Create a DataFrame with actual values and residuals
residuals_df = pd.DataFrame({
    'Actual': 1e5 * y_test,
    'Residuals': 1e5 * residuals  # Convert residuals to dollars
})

# Sort the DataFrame by actual median house values
residuals_df = residuals_df.sort_values(by='Actual')

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], 
            marker='o', alpha=0.4, edgecolor='black', color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title('Median House Value Prediction Residuals\nOrdered by Actual Median Prices')
plt.xlabel('Actual Median House Value ($)')
plt.ylabel('Residual Error ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

#Task 4. What trend can you infer from this residual plot?
# Assuming residuals and y_test are already defined
# Reuse the previously defined residuals variable
residuals = residuals * 1e5

# Create a DataFrame
residuals_df = pd.DataFrame({
    'Actual': 1e5 * y_test,
    'Residuals': residuals
})

# Bin the actual prices to observe the average residual in each bin
residuals_df['PriceBin'] = pd.cut(residuals_df['Actual'], bins=20)
bin_means = residuals_df.groupby('PriceBin')['Residuals'].mean()

# Plot the mean residuals for each price bin
plt.figure(figsize=(10, 5))
bin_means.plot(kind='bar', color='skyblue', edgecolor='black')
plt.axhline(0, color='red', linestyle='--')
plt.title('Average Residuals by Median House Price Bin')
plt.ylabel('Average Residual ($)')
plt.xlabel('Median House Price Bins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Task 5. Display the feature importances as a bar chart.
# Feature importances
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align="center", color="skyblue", edgecolor="black")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.tight_layout()
plt.show()

#Task 6. Some final thoughts to consider
# Skewness of the target
# Skewness indicates the asymmetry of the distribution of the target variable.
# A high skewness value may suggest the need for transformation to improve model performance.
target_skewness = skew(y_train)
print(f"Skewness of Median House Value: {target_skewness:.2f}")
print("A high skewness value may indicate that the target variable is not normally distributed, which could affect model predictions.")

# Count how many values are clipped at the top ($500,000 or 5.0 in target units)
# The value 5.0 corresponds to the clipped threshold of $500,000 in the dataset.
clipped_count = np.sum(y_train >= 5.0)
print(f"Number of clipped values (>= $500,000): {clipped_count}")
print(f"Percentage of clipped values: {clipped_count / len(y_train) * 100:.2f}%")

plt.hist(y_train, bins=30, color='skyblue', edgecolor='black')
plt.hist(y_train, bins=30, color='orange', edgecolor='black')
plt.axvline(5.0, color='red', linestyle='--', label='Clipped Threshold ($500,000)')
plt.title("Distribution of Median House Values (Train Set)")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

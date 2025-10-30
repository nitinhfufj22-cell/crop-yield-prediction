# crop-yield-prediction
#Predicting crop yield using weather and soil data with data analytics and visualization
# 📘 Crop Yield Prediction using Crop Production Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("crop_production.csv")

# View dataset
print(data.head())

# Drop missing values
data = data.dropna()

# Calculate yield = Production / Area
data['Yield'] = data['Production'] / data['Area']

# Remove infinite or zero values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Encode categorical columns
data_encoded = pd.get_dummies(data, columns=['State_Name','District_Name','Season','Crop'], drop_first=True)

# Select features and target
X = data_encoded.drop(['Yield','Production'], axis=1)
y = data_encoded['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plot Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()

# Visualization: Average yield by crop
avg_yield = data.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=avg_yield.values, y=avg_yield.index)
plt.title("Top 10 Crops by Average Yield")
plt.xlabel("Average Yield (Production/Area)")
plt.ylabel("Crop")
plt.show()

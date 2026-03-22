import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("FuelConsumption.csv")

# 2. Data Preprocessing
le = LabelEncoder()
df["FUELTYPE_ENC"] = le.fit_transform(df["FUELTYPE"])
df["VEHICLECLASS_ENC"] = le.fit_transform(df["VEHICLECLASS"])

# We'll use these features for the model
features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'FUELTYPE_ENC']
X = df[features]
y = df['CO2EMISSIONS']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Essential for KNN distance metrics)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Training the KNN Regressor
kn = KNeighborsRegressor(n_neighbors=5)
kn.fit(X_train_scaled, y_train)

# 6. Predictions & Evaluation
y_pred = kn.predict(X_test_scaled)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# 7. 3D VISUALIZATION SECTION
# To visualize in 3D, we pick the two most important features
viz_features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB']
X_viz = df[viz_features].values
y_viz = df['CO2EMISSIONS'].values

# Scale for the visualization model
scaler_viz = StandardScaler()
X_viz_scaled = scaler_viz.fit_transform(X_viz)

kn_viz = KNeighborsRegressor(n_neighbors=5)
kn_viz.fit(X_viz_scaled, y_viz)

# Create meshgrid for the surface
x_surf = np.linspace(X_viz[:, 0].min(), X_viz[:, 0].max(), 30)
y_surf = np.linspace(X_viz[:, 1].min(), X_viz[:, 1].max(), 30)
x_grid, y_grid = np.meshgrid(x_surf, y_surf)

# Predict over the grid
grid_combined = np.c_[x_grid.ravel(), y_grid.ravel()]
grid_scaled = scaler_viz.transform(grid_combined)
z_grid = kn_viz.predict(grid_scaled).reshape(x_grid.shape)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual points
ax.scatter(df['ENGINESIZE'], df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], 
           c='blue', alpha=0.3, label='Actual Data', s=15)

# Plot prediction surface
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='hot', alpha=0.5, linewidth=0)

ax.set_xlabel('Engine Size')
ax.set_ylabel('Fuel Consumption (Comb)')
ax.set_zlabel('CO2 Emissions')
ax.set_title('KNN Regression: 3D Prediction Surface')
plt.show()
plt.savefig('knn_3d_output.png')
print("\nVisualization saved as 'knn_3d_output.png'")
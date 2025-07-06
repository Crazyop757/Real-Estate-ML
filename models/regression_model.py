import sys
import os

# Add project root (real_estate/) to system path so visuals can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r'C:\Users\Rudra Kushwah\Documents\CODING\real_estate\data\processed\cleaned_data.csv')
data = pd.get_dummies(data, drop_first=True)

X = data.iloc[:, data.columns != 'price']
y = data.iloc[:, data.columns.get_loc('price')]

from sklearn.model_selection import train_test_split
X_train , X_test,  y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)


# linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)

# Lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 10 , random_state=42)
RF.fit(X_train,y_train)
RF_pred = RF.predict(X_test)

def evaluate(modelName , y_true , y_pred):
    print(f"\n {modelName} Evalaution")
    print("RMSE: ",round(np.sqrt(mean_squared_error(y_true,y_pred)), 2))
    print("R² Score: ", round(r2_score(y_true, y_pred), 4))


# evaluate("Linear Regression", y_test, lr_pred)
# evaluate("Ridge Regression", y_test, ridge_pred)
# evaluate("Lasso Regression", y_test, lasso_pred)
# evaluate("Random Forest Regression", y_test, RF_pred)

from visuals.model_visualization import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_feature_importance
)

# For Random Forest model results
# plot_actual_vs_predicted(y_test, RF_pred)
# plot_residual_distribution(y_test, RF_pred)
# plot_feature_importance(RF, X_train.columns)

import joblib
joblib.dump(RF, 'models/best_random_forest_model.pkl')
print("Model saved to models/best_random_forest_model.pkl")

joblib.dump(X_train.columns.tolist(), 'models/model_features.pkl')
print("Feature column list saved to models/model_features.pkl")
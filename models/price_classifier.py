import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv("data/processed/cleaned_data.csv")

model = joblib.load("models/best_random_forest_model.pkl")
features = joblib.load("models/model_features.pkl")

# Identify missing columns
missing_cols = [col for col in features if col not in data.columns]

# Create a DataFrame of missing columns (all zeros, same index as `data`)
missing_df = pd.DataFrame(0, index=data.index, columns=missing_cols)

# Concatenate all at once for performance
data = pd.concat([data, missing_df], axis=1)

# Optional: De-fragment the DataFrame
data = data.copy()

X = data[features]
y_actual = data["price"]

y_pred = model.predict(X)

price_diff = (y_pred - y_actual) / y_actual

def label_price(diff):
    return "unfair" if abs(diff) > 0.15 else "fair"

data['label'] = price_diff.apply(label_price)

fair_df = data[data['label'] == 'fair']
unfair_df = data[data['label'] == 'unfair']
fair_sampled = fair_df.sample(n=len(unfair_df)*3, random_state=42)

balanced_data = pd.concat([fair_sampled, unfair_df]).sample(frac=1, random_state=42)

X_cls = balanced_data[features]
y_cls = balanced_data['label']

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)


clf = RandomForestClassifier(class_weight={'fair': 1, 'unfair': 2}, random_state=42)
clf.fit(X_train, y_train)
y_pred_cls = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_cls))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_cls))

joblib.dump(clf, "models/price_classifier.pkl")
joblib.dump(features, "models/classifier_features.pkl")
print("Classification model & features saved")

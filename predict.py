import joblib
import pandas as pd

# === Load models and features ===
reg_model = joblib.load("models/best_random_forest_model.pkl")
cls_model = joblib.load("models/price_classifier.pkl")
features = joblib.load("models/model_features.pkl")
cls_indices = joblib.load("models/classifier_feature_indices.pkl")

# === Create empty input row ===
input_data = pd.DataFrame([0] * len(features), index=features).T

# === User input ===
input_data.at[0, 'total_sqft'] = 1200
input_data.at[0, 'bath'] = 2
input_data.at[0, 'bhk'] = 2

# Optional: if location input was included before
# selected_location = 'location_Whitefield'
# if selected_location in features:
#     input_data.at[0, selected_location] = 1

# === Predict Price ===
predicted_price = reg_model.predict(input_data)[0]
print(f"\n🏠 Predicted Price: ₹ {round(predicted_price, 2)} Lakh")

# === (Optional) Actual price input ===
# Uncomment below line to test with actual price
# actual_price = 85   # in Lakh

# Only proceed if user provides actual price
actual_price_input = input("Do you want to enter the actual price? (y/n): ").strip().lower()

if actual_price_input == "y":
    try:
        actual_price = float(input("Enter actual price (in Lakh): ").strip())
        price_diff = (predicted_price - actual_price) / actual_price

        # Reuse same logic from training
        label = "unfair" if abs(price_diff) > 0.10 else "fair"

        # Select only the classifier features
        input_cls = input_data.iloc[:, cls_indices]

        # Predict using classification model
        predicted_label = cls_model.predict(input_cls)[0]

        print(f"\n📊 Actual Price: ₹ {actual_price} Lakh")
        print(f"🔍 Price Difference: {round(price_diff * 100, 2)}%")

        if predicted_label == "fair":
            print("✅ Classified as: Fair Deal")
        else:
            print("⚠️ Classified as: Unfair Deal")
    except ValueError:
        print("❌ Invalid actual price entered.")
else:
    print("ℹ️ Skipping fairness classification since actual price was not provided.")

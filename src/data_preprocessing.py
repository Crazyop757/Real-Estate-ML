import pandas as pd

# 1. Load raw data
def load_raw_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

# 2. whitespace cleaning
def clean_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    data['location'] = data['location'].apply(lambda x: x.strip())
    return data

# 3. Extract bhk from column 'size'
def extract_bhk(data: pd.DataFrame) -> pd.DataFrame:
    data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
    return data

# 4. Convert total_sqft (handles ranges and invalid strings)
def convert_sqft_range(x):
    try:
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

def convert_sqft(data: pd.DataFrame) -> pd.DataFrame:
    data['total_sqft'] = data['total_sqft'].apply(convert_sqft_range)
    return data

# 5. Drop irrelevant columns
def drop_irrelevant_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(['society', 'availability', 'size', 'balcony'], axis=1)

# 6. Drop missing values
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()

# 7. Feature Engineering
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']
    return data

# 8. Group rare locations
def group_rare_locations(data: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    location_stats = data['location'].value_counts()
    rare_locations = location_stats[location_stats <= threshold].index
    data['location'] = data['location'].apply(lambda x: 'other' if x in rare_locations else x)
    return data

# 9. Save final output
def save_cleaned_data(data: pd.DataFrame, filepath: str) -> None:
    data.to_csv(filepath, index=False)

# 10. Master pipeline
def preprocess_data(raw_path: str, save_path: str) -> None:
    data = load_raw_data(raw_path)
    data = clean_whitespace(data)
    data = extract_bhk(data)
    data = convert_sqft(data)
    data = drop_irrelevant_columns(data)
    data = handle_missing_values(data)
    data = create_features(data)
    data = group_rare_locations(data)
    save_cleaned_data(data, save_path)

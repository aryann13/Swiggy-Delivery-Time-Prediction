"""
Swiggy Delivery Time Prediction - Data Cleaning Utilities
=========================================================
Modular cleaning functions matching notebook flow and instructor implementations.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. Standardize Column Names
# -------------------------------------------------------------------------
def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Renames all raw columns to lowercase snake_case."""
    return (
        data.rename(str.lower, axis=1)
            .rename({
                "delivery_person_id": "rider_id",
                "delivery_person_age": "age",
                "delivery_person_ratings": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken"
            }, axis=1)
    )

# Alias for compatibility
change_col_names = change_column_names


# -------------------------------------------------------------------------
# 2. Extract City Name from Rider ID
# -------------------------------------------------------------------------
def extract_city_name(data: pd.DataFrame) -> pd.DataFrame:
    """Extracts city abbreviation from rider_id (e.g. 'INDORES13DEL02' -> 'INDO')."""
    data = data.copy()
    if "rider_id" in data.columns:
        data["city_name"] = data["rider_id"].str.split("RES").str.get(0)
    return data


# -------------------------------------------------------------------------
# 3. Clean Latitude & Longitude Coordinates
# -------------------------------------------------------------------------
def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Takes absolute value of negative coordinates,
    and sets near-zero coordinates (< threshold) to np.nan.
    """
    data = data.copy()
    location_columns = [
        "restaurant_latitude", "restaurant_longitude",
        "delivery_latitude", "delivery_longitude"
    ]
    for col in location_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").abs()
            data.loc[data[col] < threshold, col] = np.nan
    return data


# -------------------------------------------------------------------------
# 4. Extract Date Features
# -------------------------------------------------------------------------
def extract_date_features(data: pd.DataFrame) -> pd.DataFrame:
    """Extracts order_day, order_month, order_day_of_week, and is_weekend from order_date."""
    data = data.copy()
    if "order_date" in data.columns:
        dates = pd.to_datetime(data["order_date"], dayfirst=True)
        data["order_day"] = dates.dt.day
        data["order_month"] = dates.dt.month
        data["order_day_of_week"] = dates.dt.day_name().str.lower()
        data["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    return data


# -------------------------------------------------------------------------
# 5. Helper Function for Time of Day
# -------------------------------------------------------------------------
def time_of_day(ser):
    return (
        pd.cut(ser, bins=[0, 6, 12, 17, 20, 24], right=True,
               labels=["after_midnight", "morning", "afternoon", "evening", "night"])
    )


# -------------------------------------------------------------------------
# 6. Extract Time Features
# -------------------------------------------------------------------------
def extract_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time features:
    - pickup_time_minutes: duration between order_time and order_picked_time
    - order_time_hour: hour of order placement
    - order_time_of_day: category (after_midnight, morning, afternoon, evening, night)
    Drops raw order_time and order_picked_time columns.
    """
    data = data.copy()
    if "order_time" in data.columns and "order_picked_time" in data.columns:
        order_time = pd.to_datetime(data["order_time"], format="mixed")
        order_picked_time = pd.to_datetime(data["order_picked_time"], format="mixed")

        data["pickup_time_minutes"] = (order_picked_time - order_time).dt.seconds / 60
        data["order_time_hour"] = order_time.dt.hour
        data["order_time_of_day"] = time_of_day(data["order_time_hour"])

        # Drop raw order time columns after extracting features
        data = data.drop(columns=["order_time", "order_picked_time"])
    return data


# -------------------------------------------------------------------------
# 7. Calculate Haversine Distance (in km)
# -------------------------------------------------------------------------
def calculate_haversine_distance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Haversine distance in kilometers between restaurant and delivery location.
    """
    data = data.copy()
    location_columns = [
        "restaurant_latitude", "restaurant_longitude",
        "delivery_latitude", "delivery_longitude"
    ]
    if all(col in data.columns for col in location_columns):
        lat1 = data[location_columns[0]]
        lon1 = data[location_columns[1]]
        lat2 = data[location_columns[2]]
        lon2 = data[location_columns[3]]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        data["distance"] = 6371 * c
    return data


# -------------------------------------------------------------------------
# 8. Create Distance Type (Exact Instructor Code)
# -------------------------------------------------------------------------
def create_distance_type(data: pd.DataFrame):
    return (
        data.assign(
            distance_type=pd.cut(data["distance"], bins=[0, 5, 10, 15, 25],
                                  right=False, labels=["short", "medium", "long", "very_long"])
        )
    )


# -------------------------------------------------------------------------
# 9. Main Notebook Data Cleaning Function
# -------------------------------------------------------------------------
def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Core data cleaning matching the notebook definition:
    - Drops 'id' column and corrupt rows (minors & 6-star ratings)
    - Replaces 'NaN ' with np.nan
    - Extracts city_name from rider_id
    - Converts numeric columns
    - Converts coordinates to absolute values
    - Extracts date features
    - Extracts time features
    - Cleans categorical columns (lowercase and strip)
    - Cleans target column (time_taken)
    """
    data = data.copy()

    # Step 1: Drop 'id' and corrupt rows
    minor_index = data[pd.to_numeric(data.get("age"), errors="coerce") < 18].index.tolist() if "age" in data.columns else []
    six_star_index = data[pd.to_numeric(data.get("ratings"), errors="coerce") == 6].index.tolist() if "ratings" in data.columns else []

    data = data.drop(columns="id", errors="ignore")
    data = data.drop(index=minor_index, errors="ignore")
    data = data.drop(index=six_star_index, errors="ignore")
    data = data.replace("NaN ", np.nan)

    # Step 2: Extract city_name from rider_id
    data = extract_city_name(data)

    # Step 3: Convert numeric columns
    numeric_cols = ["age", "ratings", "multiple_deliveries", "vehicle_condition"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Step 4: Fix location coordinates (make positive)
    location_cols = [
        "restaurant_latitude", "restaurant_longitude",
        "delivery_latitude", "delivery_longitude"
    ]
    for col in location_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").abs()

    # Step 5: Extract date features
    data = extract_date_features(data)

    # Step 6: Extract time features
    data = extract_time_features(data)

    # Step 7: Clean categorical columns
    if "weather" in data.columns:
        data["weather"] = (
            data["weather"]
            .astype(str)
            .str.replace("conditions ", "", regex=False)
            .str.lower()
            .replace("nan", np.nan)
        )

    categorical_cols = ["traffic", "type_of_order", "type_of_vehicle", "festival", "city_type"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.rstrip().str.lower().replace("nan", np.nan)

    # Step 8: Clean target column (time_taken: "(min) 24" -> 24)
    if "time_taken" in data.columns:
        data["time_taken"] = (
            data["time_taken"]
            .astype(str)
            .str.replace("(min) ", "", regex=False)
        )
        data["time_taken"] = pd.to_numeric(data["time_taken"], errors="coerce")

    return data


# -------------------------------------------------------------------------
# 10. Master Data Cleaning Pipeline (Method 1: Pipe-based)
# -------------------------------------------------------------------------
def perform_data_cleaning(data: pd.DataFrame, saved_data_path="swiggy_cleaned.csv") -> pd.DataFrame:
    """
    Executes the full pipeline via Pandas .pipe() and optionally saves to CSV:
    1. change_column_names
    2. data_cleaning
    3. clean_lat_long
    4. calculate_haversine_distance
    5. create_distance_type
    """
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
    )

    if saved_data_path:
        cleaned_data.to_csv(saved_data_path, index=False)

    return cleaned_data


# Alias
clean_data = perform_data_cleaning


# -------------------------------------------------------------------------
# Main Execution Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "swiggy.csv"
    df = pd.read_csv(DATA_PATH)
    print("swiggy data loaded successfully")
    perform_data_cleaning(df)
    print("Data cleaning completed and saved successfully!")

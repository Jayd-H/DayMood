from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def train_model(data_cleaned, features_to_exclude):
    X = data_cleaned.drop(columns=features_to_exclude)
    y = data_cleaned["Mood"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    feature_importances = model.feature_importances_
    feature_importances_percentage = (
        feature_importances / feature_importances.sum()
    ) * 100

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances_percentage}
    )

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    ).reset_index(drop=True)

    return feature_importance_df

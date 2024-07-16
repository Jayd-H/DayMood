import pandas as pd
import os


def load_and_process_data():
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Define the relative file path
    file_path = os.path.join(current_dir, "../data/data.csv")

    # Load the data
    data = pd.read_csv(file_path)

    # Convert UK date format to a proper datetime object
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")

    # Drop rows with missing values
    data_cleaned = data.dropna()

    return data, data_cleaned


def get_highest_lowest_scores(data_cleaned, features_to_exclude):
    categories = data_cleaned.columns[2:]

    highest_lowest_scores = {}
    for category in categories:
        if category not in features_to_exclude:
            highest_day_idx = data_cleaned[category].idxmax()
            lowest_day_idx = data_cleaned[category].idxmin()
            highest_day = data_cleaned.loc[highest_day_idx]["Day"]
            lowest_day = data_cleaned.loc[lowest_day_idx]["Day"]
            highest_date = data_cleaned.loc[highest_day_idx]["Date"].strftime(
                "%d/%m/%Y"
            )
            lowest_date = data_cleaned.loc[lowest_day_idx]["Date"].strftime("%d/%m/%Y")
            highest_score = data_cleaned[category].max()
            lowest_score = data_cleaned[category].min()
            highest_lowest_scores[category] = {
                "Highest Day": f"{highest_score} on {highest_date} (Day {highest_day})",
                "Lowest Day": f"{lowest_score} on {lowest_date} (Day {lowest_day})",
            }

    highest_lowest_df = pd.DataFrame(highest_lowest_scores).T

    return categories, highest_lowest_df

import sys
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
def main():
    # Increase pandas print viewport (so we see more on the screen)
    pandas.set_option("display.max_rows", 10)
    pandas.set_option("display.max_columns", 500)
    pandas.set_option("display.width", 1_000)
    # Load the famous titanic data set
    titanic_df = pandas.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )
    # Drop rows with missing values
    titanic_df = titanic_df.dropna()
    print_heading("Original Dataset")
    print(titanic_df)
    print(titanic_df.columns)
    print_heading("Only Selected Categorical Features")
    print(titanic_df[["class", "sex", "embarked", "who"]])
    # DataFrame to numpy values
    X_orig = titanic_df[["class", "sex", "embarked", "who"]].values
    y = titanic_df["survived"].values
    # Let's generate a feature from the where they started
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(X_orig)
    X = one_hot_encoder.transform(X_orig)
    # Fit the features to a random forest
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)
    test_df = pandas.DataFrame.from_dict(
        [
            {"class": "First", "sex": "male", "embarked": "C", "who": "man"},
            {"class": "Third", "sex": "male", "embarked": "C", "who": "child"},
            {"class": "Third", "sex": "female", "embarked": "C", "who": "woman"},
            {"class": "First", "sex": "female", "embarked": "S", "who": "woman"},
        ]
    )
    print_heading("Dummy data to predict")
    print(test_df)
     X_test_orig = test_df.values
    X_test = one_hot_encoder.transform(X_test_orig)
    prediction = random_forest.predict(X_test)
    probability = random_forest.predict_proba(X_test)
    print_heading("Model Predictions")
    print(f"Classes: {random_forest.classes_}")
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")
    # As pipeline
    print_heading("Model via Pipeline Predictions")
    pipeline = Pipeline(
        [
            ("OneHotEncode", OneHotEncoder()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_orig, y)
    probability = pipeline.predict_proba(X_test_orig)
    prediction = pipeline.predict(X_test_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")
    return

if __name__ == "__main__":
    sys.exit(main())

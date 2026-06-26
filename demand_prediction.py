import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Read borrowing history
data = pd.read_csv("data/borrowing_history.csv")

# Remove extra spaces
data["Month"] = data["Month"].astype(str).str.strip().str.title()
data["Book_Title"] = data["Book_Title"].astype(str).str.strip()

# Save available months
available_months = sorted(data["Month"].unique())

# Create encoders
month_encoder = LabelEncoder()
book_encoder = LabelEncoder()

# Convert text to numbers
data["Month"] = month_encoder.fit_transform(data["Month"])
data["Book_Title"] = book_encoder.fit_transform(data["Book_Title"])

# Features and Target
X = data[["Month"]]
y = data["Book_Title"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


def predict_book(month):

    # Convert user input into proper format
    month = month.strip().title()

    if month not in available_months:
        return (
            "Month not found!\n"
            f"Available months: {', '.join(available_months)}"
        )

    month_value = month_encoder.transform([month])

    prediction = model.predict([[month_value[0]]])

    book = book_encoder.inverse_transform(prediction)

    return book[0]


if __name__ == "__main__":

    print("====== Book Demand Prediction ======\n")

    while True:

        month = input("Enter Month (or exit): ")

        if month.lower() == "exit":
            print("Program Closed.")
            break

        result = predict_book(month)

        print("\nPredicted High Demand Book")
        print("--------------------------")
        print(result)
        print()
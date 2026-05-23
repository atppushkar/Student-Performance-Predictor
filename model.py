import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Dataset load
data = pd.read_csv("dataset.csv")

# Input aur Output define
X = data[["Hours_Studied", "Attendance", "Previous_Score"]]
y = data["Final_Score"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model create
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

print("Predicted Scores:")
print(prediction)

# Error
error = mean_absolute_error(y_test, prediction)

print("\nMean Absolute Error:", error)

# User input
print("\nEnter student details:")

hours = float(input("Hours studied: "))
attendance = float(input("Attendance %: "))
previous = float(input("Previous score: "))

student = [[hours, attendance, previous]]

result = model.predict(student)

print("\nPredicted Final Score:", round(result[0],2))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

dataset_path = "/Users/icentral/Desktop/TITANIC SURVIVAL PREDICTION"

df = pd.read_csv(f"{dataset_path}/Titanic-Dataset.csv")

df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

imputer = SimpleImputer(strategy="median")
df["Age"] = imputer.fit_transform(df[["Age"]])
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

table = "| Metric       | Precision | Recall | F1-Score | Support |\n"
table += "|--------------|-----------|--------|----------|---------|\n"
for index, row in report_df.iterrows():
    table += f"| {index:<12} | {row['precision']:<9.2f} | {row['recall']:<6.2f} | {row['f1-score']:<8.2f} | {row['support']:<7} |\n"

accuracy_rounded = round(accuracy, 2)
print(f"Accuracy: {accuracy_rounded:.2f}")
print(table)

joblib.dump(model, f"{dataset_path}/titanic_model.pkl")
print("Model saved as titanic_model.pkl")
 
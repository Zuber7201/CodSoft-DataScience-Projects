import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('/Users/icentral/Desktop/MOVIE RATING PREDICTION WITH PYTHON/IMDb Movies India.csv', encoding='ISO-8859-1')

print("First few rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

df['Votes'] = df['Votes'].astype(str).str.replace(r'\D', '', regex=True)  
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)

df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)

df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
df['Duration'] = df['Duration'].fillna(0)

df['Rating'] = df['Rating'].fillna(df['Rating'].median())

categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
label_encoders = {}
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown') 
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col]) 
columns_to_drop = ['Title', 'Description', 'Language', 'Country']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("\nNon-numeric columns found in df:", non_numeric_cols)
    df = df.drop(columns=non_numeric_cols)

df = df.dropna(subset=['Rating'])

X = df.drop('Rating', axis=1) 
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n| Metric                  | Value |")
print("|------------------------|----------------|")
print(f"| Mean Absolute Error (MAE) | {mae:.4f} |")
print(f"| Mean Squared Error (MSE)  | {mse:.4f} |")
print(f"| R² Score                 | {r2:.4f} |")

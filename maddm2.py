import pandas as pd
import matplotlib.pyplot as plt

# 1. Încărcarea setului de date
data = pd.read_csv('insurance.csv')

# 2. Curățarea datelor
# Verificăm valorile lipsă
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Eliminăm rândurile cu valori lipsă
data = data.dropna()

# 3. Transformarea datelor
# Transformăm coloanele categorice în variabile numerice
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# 4. Normalizarea datelor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['age', 'bmi', 'children', 'charges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Verificăm primele rânduri ale setului de date pregătit
print(data.head())

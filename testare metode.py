import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import dcc, html

# Încărcarea setului de date
data = pd.read_csv('insurance.csv')

# Curățarea datelor
data = data.dropna()

# Transformarea coloanelor categorice în variabile numerice
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Normalizarea datelor numerice, exceptând coloanele 'age' și 'charges'
scaler = StandardScaler()
numerical_features = ['bmi', 'children']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Definim caracteristicile (X) și eticheta (y) pentru clasificare
X = data.drop('smoker_yes', axis=1)
y = data['smoker_yes']

# Împărțim setul de date în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inițializăm și antrenăm modelul Random Forest
# Parametri ce pot fi modificați: n_estimators, max_depth
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluăm modelul Random Forest folosind 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rf_accuracy = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
rf_precision = cross_val_score(rf_model, X, y, cv=kf, scoring='precision')
rf_f1 = cross_val_score(rf_model, X, y, cv=kf, scoring='f1')

print(f'Random Forest - Accuracy: {rf_accuracy.mean():.4f}')
print(f'Random Forest - Precision: {rf_precision.mean():.4f}')
print(f'Random Forest - F1 Score: {rf_f1.mean():.4f}')

# Inițializăm și antrenăm modelul Logistic Regression
# Parametri ce pot fi modificați: C, max_iter
lr_model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)  # Increased max_iter to 500
lr_model.fit(X_train, y_train)

# Evaluăm modelul Logistic Regression folosind 10-fold cross-validation
lr_accuracy = cross_val_score(lr_model, X, y, cv=kf, scoring='accuracy')
lr_precision = cross_val_score(lr_model, X, y, cv=kf, scoring='precision')
lr_f1 = cross_val_score(lr_model, X, y, cv=kf, scoring='f1')

print(f'Logistic Regression - Accuracy: {lr_accuracy.mean():.4f}')
print(f'Logistic Regression - Precision: {lr_precision.mean():.4f}')
print(f'Logistic Regression - F1 Score: {lr_f1.mean():.4f}')

# Inițializăm și antrenăm modelul Support Vector Machine (SVM)
# Parametri ce pot fi modificați: C, kernel
svm_model = SVC(C=1.0, kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluăm modelul SVM folosind 10-fold cross-validation
svm_accuracy = cross_val_score(svm_model, X, y, cv=kf, scoring='accuracy')
svm_precision = cross_val_score(svm_model, X, y, cv=kf, scoring='precision')
svm_f1 = cross_val_score(svm_model, X, y, cv=kf, scoring='f1')

print(f'SVM - Accuracy: {svm_accuracy.mean():.4f}')
print(f'SVM - Precision: {svm_precision.mean():.4f}')
print(f'SVM - F1 Score: {svm_f1.mean():.4f}')

# Inițializăm și antrenăm modelul K-Nearest Neighbors (KNN)
# Parametri ce pot fi modificați: n_neighbors, weights
knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_model.fit(X_train, y_train)

# Evaluăm modelul KNN folosind 10-fold cross-validation
knn_accuracy = cross_val_score(knn_model, X, y, cv=kf, scoring='accuracy')
knn_precision = cross_val_score(knn_model, X, y, cv=kf, scoring='precision')
knn_f1 = cross_val_score(knn_model, X, y, cv=kf, scoring='f1')

print(f'KNN - Accuracy: {knn_accuracy.mean():.4f}')
print(f'KNN - Precision: {knn_precision.mean():.4f}')
print(f'KNN - F1 Score: {knn_f1.mean():.4f}')
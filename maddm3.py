import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Încărcarea și pregătirea setului de date
data = pd.read_csv('insurance.csv')
data = data.dropna()
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = ['age', 'bmi', 'children', 'charges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Definirea caracteristicilor și a etichetei
X = data.drop('smoker_yes', axis=1)  # presupunem că 'smoker_yes' este eticheta
y = data['smoker_yes']

# Împărțirea setului de date în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenarea modelului Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluarea modelului folosind 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
precision = cross_val_score(model, X, y, cv=kf, scoring='precision')
f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')

print(f"Accuracy: {accuracy.mean():.4f}")
print(f"Precision: {precision.mean():.4f}")
print(f"F1 Score: {f1.mean():.4f}")

# Modificarea parametrilor și reevaluarea modelului
model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, random_state=42)
model.fit(X_train, y_train)
accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
precision = cross_val_score(model, X, y, cv=kf, scoring='precision')
f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')

print(f"Modified Accuracy: {accuracy.mean():.4f}")
print(f"Modified Precision: {precision.mean():.4f}")
print(f"Modified F1 Score: {f1.mean():.4f}")
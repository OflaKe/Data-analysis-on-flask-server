import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
from flask import Flask, request, jsonify

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

# Inițializăm și antrenăm modelul Logistic Regression
# Parametri ce pot fi modificați: C, max_iter
lr_model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
lr_model.fit(X_train, y_train)

# Evaluăm modelul Logistic Regression
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Logistic Regression - Accuracy: {accuracy:.4f}')
print(f'Logistic Regression - Precision: {precision:.4f}')
print(f'Logistic Regression - F1 Score: {f1:.4f}')

# Creăm aplicația Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True)
    df = df.reindex(columns=X.columns, fill_value=0)
    df[numerical_features] = scaler.transform(df[numerical_features])
    prediction = lr_model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
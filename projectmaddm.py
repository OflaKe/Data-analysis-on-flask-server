import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import dcc, html

# Încărcarea setului de date
data = pd.read_csv('insurance.csv')

# Curățarea datelor
# Eliminăm rândurile cu valori lipsă pentru a asigura integritatea datelor
data = data.dropna()

# Transformarea coloanelor categorice în variabile numerice
# Convertim coloanele 'sex', 'smoker' și 'region' în variabile numerice folosind one-hot encoding
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Normalizarea datelor numerice, exceptând coloanele 'age' și 'charges'
# Standardizăm datele numerice pentru a avea media 0 și deviația standard 1
scaler = StandardScaler()
numerical_features = ['bmi', 'children']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 1. Clasificare
# Definim caracteristicile (X) și eticheta (y) pentru clasificare
X = data.drop('smoker_yes', axis=1)
y = data['smoker_yes']

# Împărțim setul de date în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inițializăm și antrenăm modelul Random Forest
# Parametri ce pot fi modificați: n_estimators, max_depth
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluăm modelul folosind 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
precision = cross_val_score(model, X, y, cv=kf, scoring='precision')
f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')

# Afișăm rezultatele evaluării
print(f'Accuracy: {accuracy.mean():.4f}')
print(f'Precision: {precision.mean():.4f}')
print(f'F1 Score: {f1.mean():.4f}')

# 2. Regresie
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 3. Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)
data['cluster'] = clusters
silhouette_avg = silhouette_score(data, clusters)

# 4. Vizualizarea datelor
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Insurance Data Visualization"),
    
    # Grafic scatter
    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(data, x='age', y='charges', color='smoker_yes', 
                          title='Scatter Plot: Age vs Charges')
    ),
    
    # Histogramă
    dcc.Graph(
        id='histogram',
        figure=px.histogram(data, x='bmi', nbins=30, color='sex_male', 
                            title='Histogram: BMI Distribution')
    ),
    
    # Box plot
    dcc.Graph(
        id='box-plot',
        figure=px.box(data, x='region_southeast', y='charges', color='smoker_yes', 
                      title='Box Plot: Charges by Region and Smoking Status')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
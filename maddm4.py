import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

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

# Aplicarea algoritmului K-means
# Inițializăm și aplicăm algoritmul K-means pentru a crea 3 clustere
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Adăugarea etichetelor de cluster la setul de date
# Adăugăm o nouă coloană 'cluster' în setul de date pentru a indica clusterul fiecărui rând
data['cluster'] = clusters

# Evaluarea performanței modelului folosind Silhouette Score
# Calculăm Silhouette Score pentru a evalua cât de bine sunt separate clusterele
silhouette_avg = silhouette_score(data, clusters)
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Vizualizarea clusterelor
# Creăm un grafic scatter pentru a vizualiza clusterele pe baza vârstei și a costurilor
plt.scatter(data['age'], data['charges'], c=data['cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Clustering of Insurance Data')
plt.colorbar(label='Cluster')
plt.show()
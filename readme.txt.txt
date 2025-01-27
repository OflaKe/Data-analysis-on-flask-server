S-au incercat mai multe metode de clasificare a datelor aceasta prezentata fiind cea mai potrivita pentru acest MLM (LR).

Random Forest - Accuracy: 0.9544
Random Forest - Precision: 0.8669
Random Forest - F1 Score: 0.8892
Logistic Regression - Accuracy: 0.9559
Logistic Regression - Precision: 0.8703
Logistic Regression - F1 Score: 0.8957
SVM - Accuracy: 0.9454
SVM - Precision: 0.7906
SVM - F1 Score: 0.8773
KNN - Accuracy: 0.9230
KNN - Precision: 0.8071
KNN - F1 Score: 0.8119

Pregătirea datelor:

Am încărcat setul de date insurance.csv și am eliminat rândurile cu valori lipsă.
Am transformat coloanele categorice (sex, smoker, region) în variabile numerice folosind one-hot encoding.
Am normalizat coloanele bmi și children folosind StandardScaler.
Clasificare:

Am folosit un model Random Forest pentru a prezice statutul de fumător (smoker_yes).
Parametri ce pot fi modificați: n_estimators (numărul de arbori în pădurea aleatorie), max_depth (adâncimea maximă a fiecărui arbore).
Am evaluat modelul folosind 10-fold cross-validation și am obținut metrici de performanță precum acuratețea, precizia și scorul F1.
Regresie:

Am folosit un model de regresie liniară pentru a prezice costurile (charges).
Am evaluat modelul folosind eroarea pătratică medie (MSE).
Clustering:

Am aplicat algoritmul K-means pentru a crea 3 clustere.
Parametri ce pot fi modificați: n_clusters (numărul de clustere).
Am evaluat performanța modelului folosind Silhouette Score.
Vizualizarea datelor:

Am creat un raport interactiv folosind Dash și Plotly, incluzând un grafic scatter, o histogramă și un box plot pentru a vizualiza relațiile și distribuțiile cheie din setul de date.
Observații
Scatter Plot: Relația dintre vârstă și costuri arată că fumătorii tind să aibă costuri mai mari, indiferent de vârstă.
Histogramă: Distribuția BMI arată că există o variație semnificativă între bărbați și femei.
Box Plot: Costurile medicale sunt mai mari pentru fumători în toate regiunile, cu variații semnificative între regiuni.
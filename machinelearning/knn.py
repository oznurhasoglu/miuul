################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

#uzaklık temelli bir yöntem olduğu için standartlaştırma yapıyoruz.
X_scaled = StandardScaler().fit_transform(X)

# x_scaled np arraytipinde ve sütun isimlerini taşımıyor, sütunları ekliyoruz ve dataframe olarak kaydediyoruz.
X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1] # bağımsız değişkenlerin 1 sınıfına ait olma olasılıkları. bunun üzerinden roc auc hesaplıycaz.

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob)
# 0.90

# cross_val_score() metodunda tek metriğe göre değerlendirilir. cross_validate() birden çok metriğe göre değerlendirme yapabilir.
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() # 0.73
cv_results['test_f1'].mean() # 0.59
cv_results['test_roc_auc'].mean() # 0.78

# Başarıyı arttırmak için aşağıdakiler yapılabilir:
# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

# hiperparametrelerine bakalım
################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

#default değeri 5 olan komşuluk sayısı parametresini 2 ile 50 arasında değerler alsın hepsi de denensin istiyorum.
#bu işlemi grid search algoritması ile yapabilirim

knn_params = {"n_neighbors": range(2, 50)}

# hiperparametre optimizasyonu yaparen de cross val kullanılır. cv
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1, # işlemciyi tam performansla kullanır -1 olunca
                           verbose=1).fit(X, y) # verbose yapılan işlem için rapor istiyor musun diye sorar

# en iyi parametreler neymiş diye bakıyorum. 17 imiş.
knn_gs_best.best_params_

################################################
# 6. Final Model
################################################

# grid search sonucu elde ettiğim parametreleri tek tek yazmak yerine
# set_params metoduyla grid search çıktımı başına ** koyarak yazdığımda değişkenleri oraya atar.
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #76
cv_results['test_f1'].mean() #61
cv_results['test_roc_auc'].mean() #81
# sonuçlarımız iyileşti

# bir tahmin yapalım
random_user = X.sample(1)

knn_final.predict(random_user)

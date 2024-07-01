######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x) #virgülden sonra 2 basamak göster

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.shape

# şimdilik iki değişken alıp arasındaki doğrusal ilişkiye bakalım.
X = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################

# modeli tanımladık, x ve y'yi verdik
reg_model = LinearRegression().fit(X, y)

# fonksiyonumuz şöyleydi:
# y_hat = b + w*TV

# sabit (b - bias)
reg_model.intercept_[0] #modelin bulduğu sabit

# tv'nin katsayısı (w1)
reg_model.coef_[0][0] # modelin değişkenler için bulduğu katsayılar


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

# modelin bulduğu parametreleri yerine koyarak tahmin yapabiliyorum.
reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500



# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r") #ci: güven aralığı
# grafiğin titleını dinamik yapıyoruz yani model her güncellendiğinde başlıktaki b ve w değerleri de değişecek.
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310) # grafiğin başlangıç ve bitiş noktaları
plt.ylim(bottom=0)
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X) #train test vs ayırmadığım için şimdilik sadece eldeki veriyi tahmin ettiriyorum.
mean_squared_error(y, y_pred)
# 10.51
y.mean() # mse değeri düşük mü yüksek mi bilmek için bağımlı değişkenin ort. ve ss'ına bakıyorum.
y.std()

# RMSE: üsttekinin kökünü aldım
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE: bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir. değişken sayısı arttıkça şişmeye meyillidir.
reg_model.score(X, y)
#%61

######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1)

y = df[["sales"]]


##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90 ---> b değeri
# 0.0468431 , 0.17854434, 0.00258619 --->w değerleri

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 #6.20
# şimdi modelimize tahmin ettirelim
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T # verinin değerlerini dataframe olarak tanımlıyorum

reg_model.predict(yeni_veri) # modele veriyorum. az önce benim bulduğumla aynı sonucu bulacak 6.20

##########################
# Tahmin Başarısını Değerlendirme
##########################

# train seti üstünde metriklere bir bakayım
# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train)
#0.89

# bir de test seti üstünde metriklere bir bakayım. arada uçuk farklar varsa ve train testten aşırı iyiyse overfit vardır.
# bunda yok aksine çok iyi
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test RKARE
reg_model.score(X_test, y_test)

# değişken sayısı artınca sonuçlar iyileşti haliyle.

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# cross_val_score() fonksiyonunda scoring olarak neg_mean_squared_error verildiği için bu fonksiyonun çıktısı negatif değerler olur.
# pozitifini görmek için başta - ile çarptık. 10adet nmse değeri elde ettik. bunların np.sqrt() ile karekökünü aldık rmse elde ettik.
# son olarak ortalamasını aldık.

# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71

# cv' nin kaç katlı olacağına veri setinin boyutuna göre karar verebilirsin.


# kendimiz yazalım modeli baştan pekiştirme amaçlı
######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y) # gözlem sayısını tutuyoruz
    sse = 0 # hata kareleri toplamı

    for i in range(0, m): # tüm gözlem birimlerinde gez
        y_hat = b + w * X.iloc[i] # bağımsız değişkenin y değerini tahmin ediyoruz/hesaplıyoruz.
        y = Y.iloc[i] # gerçek y değerine bakıyoruz
        sse += (y_hat - y) ** 2 # sse toplamına hatanın karesini ekliyoruz

    mse = sse / m # hataların kareleri toplamının ortalamasını alarak mse değerini buluyoruz.
    return mse


# update_weights tek iterasyonluk fonksiyon. tek bir ağırlık çifti için (b,w)
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0 # hesaplayacağım türevler toplamını tutmak için
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X.iloc[i]
        y = Y.iloc[i]
        b_deriv_sum += (y_hat - y) # formüle göre kısmi türevi hesaplıyoruz hem b hem w için.
        w_deriv_sum += (y_hat - y) * X.iloc[i] # her gözlem için çıkan türevleri topluyoruz. en son ortalamasını alacağız.
    new_b = b - (learning_rate * 1 / m * b_deriv_sum) # eski değerden türevlerin ortalamasının lr içe çarpımı kadar inerek yeni değeri buluyoruz.
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu: üstteki fonksiyonlarımızı belirli bir iterasyonda kullanarak train etme işlemi
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    # ilk ağırlık değerleriyle GD sonucu mse değerimi yazdırıyorum ki sonradan karşılaştırma yapabileyim.
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters): # her iterasyon için ağırlıkları güncelleyip cost hesaplıyor ve cost historye ekliyorum.
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        # 100 iterasyonda bir raporlama yap.
        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

# şimdi bunları kullanalım.
df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters örnek olarak verdik
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
#16000. iterasyondan sonra hata düşmüyor sabit kalıyor.

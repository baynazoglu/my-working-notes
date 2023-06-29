#bagımlı degıskenle bagımsız degıskenler arasındaki ilişkiyi dogrusal sekilde temsil etmeye calısmak.
# Yi = b + wxi
# b = beta, bayes, sabit olarak karsımıza gelebilir.
# w = katsayı,wet,coeff egim bilgisi.

# Yi = b + w1x1 + w2x2 + .... wnxn...
#örnegin ev fiyat degerleme veri setimiz olsun. bagımsız degiskenler x. onların bir katsayısı var o da w. mesela
# m2 degiskeni(orn x1 diyelim) arttıkca Yi yani bagımlı degisken fiyat artacak mı? evet. o zaman w1 degeri + deger.
#orn bina yasi degiskeni(x2) arttıkca Yi azalacak. o zaman w2 nego bir degisken  vs.
#peki bu katsayıları w1,w2 nasıl bulunacak?

# ornegin m2 v fiyat grafiği olsun. bu değerler için çizilen bir doğrunun denklemi Yi = b +wxi. w egimi b katsayı.
#bu dogruyu nasıl cizeceğiz ve nereden? işte bu temel soru. DOGRUNUN B VE W DEGERLERINE RANDOM BI DEGER VERIRIZ MESELA
# SONRA CIKAN DENKLEMLE Yi hesaplanır. Hesaplanan Yi- gerçek Yi bizim hatamızı verir. MSE la hata karaler ortalamasına bakarız
# iterasyonla MSE değerinin en optimal haline ulaşmaya çalışırız. en optimaldeki b ve w degerleri bizim için uygun katsayılardır.
#
##Regresyon Modellerinde Başarı Değerlendirme.
#RMSE, MSE, MAE. tabi bunlar bize kalmış istediğimizi kullanabiliriz rmse mse ye göre daha az deger verdi onu kullanayım
# hayır. o kendi içinde bir ölçüt. verdiğin katsayılar arasında karşılaştırma yap ya da rmse içinde kendi kendine...


##Parametrelerin(ağırlıkların) katsayıların, tahmincilerin (hepsi aynı) bulunması.

#b ve w yani.
#Cost(b,w) = MSE = 1/2m ∑((b+wxi)- yi)^2    b+wxi zaten tahmin ettigimiz Yidi. yani mse yazdık yine.

######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/advertising.csv")

df.head()
df.shape

#1bagımlı 1 bagımsızla işlem yapacagız. simple cünkü.

X = df[["TV"]]
y = df[["sales"]]

##########################
# Model
##########################

reg_model = LinearRegression().fit(X,y)
#formula is : y_hat = beta + w * X(tv buradaki x)

#sabitimiz, betamız, biasımız

reg_model.intercept_[0]

#Xin katsayı:
reg_model.coef_[0][0]

##########################
# Tahmin
##########################

#mulakatlarda model denkleminin yazılması sorulur...
#yi = b +wixi + wnxn ....
#formula:
reg_model.intercept_[0] + reg_model.coef_[0][0] * X

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500


df.describe().T

#gorduk ki tv max deger 296. biz 500e baktık veride 500ler yok ama artık örüntüyü ögrendigi icin tahmin yapabildi.


# Modelin Görselleştirilmesi

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("Tv Harcalamaları")
plt.xlim(-10,310)
plt.ylim(bottom = 0)
plt.show(block = True)


##########################
# Tahmin Başarısı
##########################

#mse

y_pred = reg_model.predict(X)

mean_squared_error(y,y_pred)
#10.51 cıktı. ort satısın 14, std. sap. 5 oldugu bi yerde [9,19] olan yerde
# 10.51 cok yüksek bir hata.


#rmse

np.sqrt(mean_squared_error(y,y_pred))
#3.24

#mae

mean_absolute_error(y,y_pred)


#r kare

reg_model.score(X, y)
#r kare demek x degiskeninin Y yi ne kadar temsil ettiği.

######################################################
# Multiple Linear Regression
######################################################

df.head()

X = df.drop("sales", axis = 1)
y = df[["sales"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)

X_train.shape
X_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)

#sabitimiz.
reg_model.intercept_

#katsayılarımız...
reg_model.coef_

##########################
# Tahmin
##########################
# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# Sales = 3.05  + TV * 0.04735257 + radio * 0.17323832 + newspaper * 0.00466519

3.05  + 30 * 0.04735257 + 10 * 0.17323832 + 40 * 0.00466519
#6.38 cıktı.

#ya da:

yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE

y_pred = reg_model.predict(X_train)

np.sqrt(mean_squared_error(y_train,y_pred))
#1.5713

#TRAIN RKARE

reg_model.score(X_train,y_train)
#%91

#TEST RMSE:

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#testteki sonucumuz 2.12 cıktı. hata testte daha buyuk...

#10 katlı Cross Validation RMSE:

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

#1.69


#5 katlı CV Rmse
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71. 5 katlı denedik cünkü verideki degerler azdı 10 cok olur diye düsündük. ama 5le 10 arasında cok bi şey degismedi..


######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

#Scikit Learn Gradient Descent kullanmıyordu. onun yerine simple normal kullanıyordu. yukarıdakilerden goruldugu uzere. (y_hat = b + wixi)
#bu sebeple Gradient Descenti biz burada yazıp sonucları gorecegiz. bu bolum bonustur. kavramak acısından yazdık...


# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]
#HYPERPARAMETRE: VERISETINDE OLMAYAN KULLANICIN DISARIDAN VERDIGI DEGERLERDIR.
# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)



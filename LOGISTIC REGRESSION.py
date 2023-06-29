#amac: sınıflandırma problemleri için bağımlı ve bağımsız değişkenler arasındaki ilişkiyi doğrusal olarak modellemektir.

######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("DERSLER/MAKINE OGRENMESI/datasets/diabetes.csv")
df.head()
##########################
# Target'ın Analizi
##########################
df["Outcome"].value_counts()

sns.countplot(x="Outcome",data = df)
plt.show(block = True)

100 * df["Outcome"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show(block=True)

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df,col)

#bagımlı degiskeni cıkarmak istersek.

cols = [col for col in df.columns if "Outcome" not in col]

cols

df.describe().T
#bazı degerler 0 ama 0 olamaz. demekki nullları 0la doldurmuslar.

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"})
#hepsi icin yapalım..

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################

df.shape
df.isnull().sum()
#yok.

for col in cols:
    print(col,check_outlier(df,col))
#insulinde outlier var.

replace_with_thresholds(df,"Insulin")

#standartlastırma yapalım...

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

######################################################
# Model & Prediction
######################################################

y = df["Outcome"]
X= df.drop(["Outcome"],axis =1)

log_model = LogisticRegression().fit(X,y)

log_model.intercept_ #katsayımız
log_model.coef_ #w degerleri...

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

######################################################
# Model Evaluation
######################################################
#en basta kendimiz sekil bir func yazalım...

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show(block=True)

plot_confusion_matrix(y,y_pred)

#pythonda olan ise:

print(classification_report(y,y_pred))
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

#ROC AUC a bakalım....

y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y,y_prob)
#0.8393
######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=20,
                                                    random_state=17)

log_model = LogisticRegression().fit(X_train,y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test,y_pred))

#burada degeler daha iyi geldi. ama random statete random bi deger atadık, ya diğerleri?

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show(block=True)
#0.85 auc degeri
# AUC
roc_auc_score(y_test, y_prob)
 #0.85


######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results

#{'fit_time': array([0.02846146, 0.00700355, 0.00699663, 0.00599957, 0.00800204]),
# 'score_time': array([0.00800085, 0.01199961, 0.00900269, 0.00900412, 0.00799942]),
 #'test_accuracy': array([0.77272727, 0.74675325, 0.75324675, 0.81699346, 0.77124183]),
# 'test_precision': array([0.71111111, 0.64705882, 0.71052632, 0.79069767, 0.73684211]),
# 'test_recall': array([0.59259259, 0.61111111, 0.5       , 0.64150943, 0.52830189]),

#5 test yaptı 5 için de degerler verdi bunların ort. alalım.




cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327


######################################################
# Prediction for A New Observation
######################################################

#yeni bir deger geldi ve onun tahminini soruyorlar...

X.columns

#tek tek deger eklemek yerine random user secelim....

random_user = X.sample(1,random_state=45)

log_model.predict(random_user)
#1 cıktı demekki bu userı diabet olarak tahmin ediyoruz...

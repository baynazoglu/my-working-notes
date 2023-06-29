#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
from  sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def load_application_train():
    data = pd.read_csv("DERSLER/Feature Engineering/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()

#böylece tekrar ilk veri setini çat diye yükleyeceğiz.

def load_titanic():
    data = pd.read_csv("DERSLER/Feature Engineering/datasets/titanic.csv")
    return data

df = load_titanic()
df.head()

#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################
###################
# Grafik Teknikle Aykırı Değerler
###################
#boxplot aykırı değer görmek için uygun bir grafiktir.

sns.boxplot(x=df["Age"])
plt.show(block= True)

#gördük ki 62 =~ 0.75 quartile. bunun üstüne aykırı deger gibi..

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up =  q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"]< low) | (df["Age"] > up)]
#aykırı değerlerimiz....
df[(df["Age"]< low) | (df["Age"] > up)].index
#aykırı değerlerimizin indexleri...

###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"]< low) | (df["Age"] > up)].any(axis=None)
#aykırı değer var mı ? any ile sorduk var dedi.

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

def outlier_thresholds(dataframe, colname, q1 =0.25, q3= 0.75):
    quartile1 = dataframe[colname].quantile(q1)
    quartile3 = dataframe[colname].quantile(q3)
    interquartile = quartile3 - quartile1
    up_limit = quartile3 + interquartile * 1.5
    low_limit = quartile1 - interquartile * 1.5
    return  low_limit, up_limit
#aykırı degerleri belirledik

outlier_thresholds(df,"Age")
outlier_thresholds(df,"Fare")
df[(df["Fare"] < low) | (df["Fare"] > up)].index

low, up = outlier_thresholds(df,"Age")

def check_outlier(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    if dataframe[(dataframe[colname] < low ) | (dataframe[colname] > up)].any(axis=None):
        return True
    else:
        return False
#burada aykırı değer var mı diye kontrol ettik veri setinin istenilen columnunda.


check_outlier(df,"Age")

###################
# grab_col_names
###################

#daha önce de yazmıstık bu functionu. veri setindeki columnların
# hangi veri tipine ait oldugunu bulma işi. görünürde cat,int olabilir ama bazıları
# cat gibi gözüküp inttir. bazıları cat gözüküp cardinaldir vs. onları ayıracağız.


def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    """

       Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
       Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

       Parameters
       ------
           dataframe: dataframe
                   Değişken isimleri alınmak istenilen dataframe
           cat_th: int, optional
                   numerik fakat kategorik olan değişkenler için sınıf eşik değeri
           car_th: int, optinal
                   kategorik fakat kardinal değişkenler için sınıf eşik değeri

       Returns
       ------
           cat_cols: list
                   Kategorik değişken listesi
           num_cols: list
                   Numerik değişken listesi
           cat_but_car: list
                   Kategorik görünümlü kardinal değişken listesi

       Examples
       ------
           import seaborn as sns
           df = sns.load_dataset("iris")
           print(grab_col_names(df))


       Notes
       ------
           cat_cols + num_cols + cat_but_car = toplam değişken sayısı
           num_but_cat cat_cols'un içerisinde.
           Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

       """
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" ]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

grab_col_names(df)
df.dtypes
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#passengerId index gibi bi şey. onu cıkaralım.
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df,col))
#num colsların aykırı değeri var mı diye check ettik.

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outliers(dataframe, colname, index = False):
    low, up = outlier_thresholds(dataframe,colname)

    if dataframe[((dataframe[colname] < low) | (dataframe[colname] > up))].shape[0] > 10:
        print(dataframe[((dataframe[colname] < low) | (dataframe[colname] > up))].head())
    else:
        print(dataframe[((dataframe[colname] < low) | (dataframe[colname] > up))])

    if index:
        outlier_index = dataframe[((dataframe[colname] < low) | (dataframe[colname] > up))].index
        return outlier_index

#outliers olanları yazdırdı.
grab_outliers(df,"Age", True)

outlier_thresholds(df,"Age")
check_outlier(df,"Age")
grab_outliers(df,"Age",True)

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df,"Age")

df[~((df["Age"] < low) | (df["Age"] > up))].shape
#Tilda kullanarak thresholdları attık.

def remove_outliers(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    df_without_outliers = dataframe[~((dataframe[colname] < low) | (dataframe[colname] > up))]
    return df_without_outliers

remove_outliers(df,"Age")

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    df_without_outliers = remove_outliers(df,col)

df.shape    #891
df_without_outliers.shape #775

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

low,up = outlier_thresholds(df,"Age")
df[((df["Age"] < low) | (df["Age"] > up))]["Age"]
#outlierslar geldi.
df.loc[((df["Age"] < low) | (df["Age"] > up)),"Age"]
#aynı işlem locla yapılanı.
df.loc[(df["Age"] > up),"Age"] = up

df.loc[(df["Age"] < low),"Age"] = low

def replace_with_thresholds(dataframe,colname):
    low, up = outlier_thresholds(dataframe,colname)
    df.loc[(df[colname] > up), colname] = up
    df.loc[(df[colname] < low), colname] = low


df = load_titanic()

cat_cols,num_cols,cat_but_car = grab_col_names(df)
for col in num_cols:
    print(col,check_outlier(df,col))
for col in num_cols:
    replace_with_thresholds(df,col)
for col in num_cols:
    print(col,check_outlier(df,col))

#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################
#LOF YAPACAGIZ
#bura biraz bonus gibi. yüksek seviye yazılım denebilir.
#bir değer tek başına outliers gözükmeyebilir ama başkasıyla miksleyince outliers olabilir
# mesela: 17 yaş bir outlier mi? hayır. 3 çocuğunun olması outlier mi? hayır. ama 17 yaşında 3 çocuğu varsa bu outlier.

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=['float64', 'int64'])
#veri cok buyuk sadece intleri sectik ki işlemleri onlarda yapacagız.
df = df.dropna()
df.shape
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

#hepsinde var gözüküyor.
#carat'a yogunlasalım.
low,up = outlier_thresholds(df,"carat")
df[((df["carat"]<low) | (df["carat"]> up))].shape
#1889 tane outliers var

low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape
#2545 tane outliers var.

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
#mantık şu: bir fit yapıyor ve 1den uzaklaşan her deger outlierslıkla dogru orantılıdır. 1.2, 8 den cok daha az outlierstir
# karar senin istedigini outliers seçebilirsin.

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show(block= True)
#grafiğe bakınca -4ten sonrası normalleşiyor ama -4 daha küçük değerler felaket outliers duruyor.
th = np.sort(df_scores)[3]
#bi tane threshold degeri atadık.
df[df_scores < th]

df[df_scores < th].shape


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
#ağaç yöntemlerinde bu outlierslar cok onemli degil. fakat dogrusal yontemlerde onemlidir.


#############################################
# Missing Values (Eksik Değerler)
#############################################
#############################################
# Eksik Değerlerin Yakalanması
#############################################
df = load_titanic()
df.head()

#eksik gozlem var mı yok mu?
df.isnull().values.any()

#degiskenlerdeki eksik deger sayıları
df.isnull().sum()

#veri setindeki toplam eksik sayısı:
df.isnull().sum().sum()

#en az 1 tane eksik değere sahip olan gözlem birimleri:

df[df.isnull().any(axis=1)]

#tam olanlar:
df[df.notnull().any(axis=1)]

#azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending = False)

#yüzde kaçı boş ?

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)


#boş olan columnları bir listede toplamak...
na_cols = [col for col in df.columns if df[col].isnull().sum()> 0]


def missing_values_table(dataframe,na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)
###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################
#meanle medianla dooldurma işi.

df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())
df["Age"].fillna(0)

#bunlar intler için geçerli objecti meanle dolduramayız.

df.apply(lambda x: x.fillna(x.mean() if x.dtype != "O" else x, axis = 0))

dff = df.apply(lambda x: x.fillna(x.mean() if x.dtype != "O" else x, axis = 0))

dff.isnull().sum()
#cabin ve embarkedda hala var cunku onlar object.

df["Embarked"].fillna(df["Embarked"].mode())
#kategoriklerde genelde modu alınır.

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
#x object ve unique sayısı 10dan azsa  modeunu al. öbür türlü isim gibi bi değişkene mod atamak sacma olur.


###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df.groupby("Sex")["Age"].mean()
df["Age"].mean()
#burada cinsiyete göre ortalamaya baktık.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))

# ya da
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"),"Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma(knn mesela)
#############################################

df = load_titanic()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
#sex ve embarked categorikdi. onları int e cevirdi.
dff.head()

#degiskenleri standartlıştıralım.

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff),columns=dff.columns)
dff.head()
#0la 1 arası oldu bütün veri seti.

#knn uylaması.

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform((dff)),columns = dff.columns)

dff.head()
#knn basit olarak der ki, bana arkadaşını söyle kim olduğunu söyleyeyim. boş değeri ona en yakın 5 arkadaşının ortalamasına gore doldururr.

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]

#yani burada null değerle knnin atadıgı degere baktık.

###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma

#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df)
plt.show(block=True) #eksik degerlerin grafiğini yaptık

msno.matrix(df)
plt.show(block=True)
#burada eğer columnların grafiği aynıysa, eksik değerleri aynı satırdadır gibi bir sonuç çıkabilir.

msno.heatmap(df)
plt.show(block=True)

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################
df = load_titanic()
missing_values_table(df)
na_cols = missing_values_table(df,True)

def missing_vs_target(dataframe,target,na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(),1,0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df,"Survived",na_cols)
#ornegin: cabin_na_flag 0: null olmayan. bunun bağımlı değişken("Survived") oranı: 0.667,
#ama cabin_na_flag: 1 yani null olanlar. bunun bağımlı değişken("Survived") oranı: 0.300. burada bi sıkıntı var.
#null values bağımlı değişkene bağlı.
#öğrendik ki titanicte çalışanların cabin numarası yokmuş. ve çoğu ölmüş.direkt bu na leri çıkarsaydık bu bilgiye ulaşamayacaktık.
#bu gelişmiş analiz biraz bonus gibiydi...ileri seviyeydi.

###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)

#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################
#degiskenlerin temsil sekillerini değiştirme işlemi.
#############################################
# Label Encoding & Binary Encoding
#############################################

df = load_titanic()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]# alfabe sıralamasına gore ılk 0 digeri 1
le.inverse_transform([0,1]) #0 ve 1 in ne oldugunu görmek istersek

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load_titanic()
label_encoder(df,"Sex").head()

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
               and df[col].nunique()== 2 ]

for col in binary_cols:
    label_encoder(df,col)
df.head()
df["Embarked"].value_counts() #3 deger var.
df["Embarked"].nunique() #3 degeri yazdı.
len(df["Embarked"].unique()) # 4 dedi. null u da 1 deger saydı.

#############################################
# One-Hot Encoding
#############################################
# S,Q,C degerleri var. aralarında bir hiyerarşi yok
#Bu sebeple bunlara 0,1,2 degerleri vermek aralarında bir sınıf farkı varmış gibi algılanılır
# Label encoding yapmak buna uygun degil. Buna yapmamız gereken şey one hot encoding.
df = load_titanic()

df["Embarked"].value_counts()

pd.get_dummies(df,columns=["Embarked"]).head()

pd.get_dummies(df,columns=["Embarked"],drop_first=True).head()

pd.get_dummies(df,columns=["Sex","Embarked"],drop_first=True).head()
#Ilk sınıfı drop edersek birbiri üzerinden oluşturma işi iptal edilmiş olur
#Dummy degisken yazarken drop_first etmemiz gerekir ki, fazla column yazılmasın
# Cünkü digerleri oldugu icin zaten cıkarım yapabilirsin..

def one_hot_encoder(dataframe,categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10>= df[col].nunique() > 2]

one_hot_encoder(df,ohe_cols).head()
#############################################
# Rare Encoding
#############################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df,col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["SibSp"].value_counts()
df.groupby("SibSp")["Survived"].mean()

def rare_analyser(dataframe,target,cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df,"Survived",cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "Survived", cat_cols)

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load_titanic()
ss=StandardScaler()
df["Age_StandartScaler"] = ss.fit_transform(df[["Age"]])
df.head()

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_RobustScaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

mms = MinMaxScaler()
df["Age_MinMaxScaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

age_cols = [col for col in df.columns if "Age"in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


        for col in age_cols:
            num_summary(df,col,True)

#burada gördük ki age ve diğer bizim stand. yaptıgımız columnların dagılımı aşağı yukarı aynı.

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)
df.head()
#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load_titanic()
df.head()
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")
#cabinde dolu olanları 1 olmayanları 0 gosterecek yani true ve false döndürecek.

df.groupby("NEW_CABIN_BOOL").agg({"Survived":"mean"})

#gozle gorulur bir farkımız var. null olanların yasam oranı 0.3. digerinin 0.667
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#pval 0.0000... so H1 rejected. yani eşitlik yok. önermemiz dogru.

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})
#burada da bi fark gözüküyor. test edelim.


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#pval 0.0000... so H1 rejected. yani eşitlik yok. önermemiz dogru.

############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################
df.head()

###################
# Letter Count
###################
df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})
df.head()

###################
# Regex ile Değişken Türetmek
###################
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("DERSLER/Feature Engineering/datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date

dff.head()
#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
df = load_titanic()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()

#korkunç bi fark var. seniorfemale vs maturemale arasında.


import numpy as np
var1 = [-1,-3,4,5,2,-6]

var2  = [0,-1,0,-3,9,4]

var3 = [0,3,45,78,21,5]

var4  = [45,27,4,4,2.34]
np.log(var3)

#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load_titanic()
df.head()
df.isnull().sum()

#column isimlerini hepsini büyük harf yapalım. büyük kücük önemli

df.columns = [col.upper() for col in df.columns]
df.columns

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

#cabin bool a bakalım. cabin nu olup olmama.
# cabin no olmayanlar crewdendi ona bakmısmtık onemlı.

df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype("int")
#inte cevirince true=1 oldu.

#name count

df["NEW_NAME_COUNT"] = df["NAME"].str.len()

#name word count

df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

#name dr. dr olanları bi columna koyalım.

df["NEW_NAME_DR"] = df["NAME"].apply(lambda x : len([x for x in x.split() if x.startswith("Dr")]))

#name title.

df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

#family size

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

#age * pclass

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

#is alone olma durumu...

df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"

#age level...

df.loc[(df["AGE"]< 18), "NEW_AGE_CAT"] ="young"
df.loc[(df["AGE"]>= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] ="mature"
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] ="senior"

df.head()

#sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()
df.shape #12column vardı biz 10 tane daha ekledik.

cat_cols,num_cols,cat_but_car = grab_col_names(df)
#passengerid istemiyoruz. herkese özgün bi ayırım o cunku.

num_cols = [col for col in num_cols if "PASSENGERID" not in col ]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col,check_outlier(df,col))
#4  columnda da varmış

for col in num_cols:
    replace_with_thresholds(df,col)
#thresholdları degistirdik.

for col in num_cols:
    print(col,check_outlier(df,col))
#false oldular sımdı. degistirdik.


#############################################
# 3. Missing Values (Eksik Değerler)
#############################################
df.columns
missing_values_table(df)
#cabin embarked age ve age e baglı olusturdugumuz degerler var.
#cabini zaten new_Cabin_bool yaptık true false. orj ihtiyac yok.
df.drop("CABIN",inplace=True,axis=1)

#ticket ve name zaten istenmedik columnlardı.

remove_cols = ["TICKET","NAME"]

df.drop(remove_cols, inplace = True, axis = 1)
#boş ageleri tittledakilerin ve cinsiyet kırımına göre ortalamasıyla dolduralım.
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

#age e baglı olarak atadıgımız tüm new columnları tekrar yapalım.

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.isnull().sum()

#embarkedda 2 boş kaldı geri kalanı zaten notnull.

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype== "O" and len(x.unique())<= 10) else x, axis = 0)

df.isnull().sum()

#modunu aldık 10dan kucuk unique varsa. öbür türlü zaten hiçbir şey yapamayacagımız şeye geliyor. tıpkı name gibi. silerdik.

#############################################
# 4. Label Encoding
#############################################
#label encoding aynı zamanda binary encoding demekti. categorical ya da binary olan degerleri int e cevirecegiz ama 2 degeri olmalı toplam ki 0 veya 1 yapalım.
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df,col)

df.columns
df.head()

#############################################
# 5. Rare Encoding
#############################################
#cat columnlardaki rareları ayıklayalım bi neymiş bakalım.
rare_analyser(df,"SURVIVED",cat_cols)

df=rare_encoder(df,0.01)

df["NEW_TITLE"].value_counts()
#rare olanları rare diye bir degisken atadı.


#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

#2den buyuk 10dan kucuk olan columnları one hot encoding cols a atadık.

df= one_hot_encoder(df,ohe_cols)
df.head()

df.shape
#12 idi 49 tane columns yaptık...

#tekrar bakalım bu 49 tanenin kaçı cat, kacı num...

cat_cols,num_cols,cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df,"SURVIVED",cat_cols)
#rare da bakıyoruz bazıları cok gereksiz. one hot encoder yaptık mesela,
# new_fam_sizeı 12 column yaptı 1,2,3. bunun gibi cok fazla gereksizler var. orn: new_fam sizeın 1 valueları hep 5-6 falan. gereksizzzz.
#bunlar cok gereksiz olanlar.  silebiliriz, kaladabilir.

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


#df.drop(useless_cols, axis=1, inplace=True)
#silmek icin yukarıdakini kullancagız ama bize baglı silip silmemek...


#############################################
# 7. Standart Scaler
#############################################

#bu uygulama için gerekli olmasa da, çoğu uygulamada gereklidir. burada zaten valuelar 0 ya da 1.

df.head()

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

#############################################
# 8. Model
#############################################

y = df["SURVIVED"] #bagımlı degisken
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1) #bagımsız degsikenler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#0.80 YANİ YUZDE 80 BASARI ORANLI BIR ML YAPTIK
#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load_titanic()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#0.70 EN BASIC FEATURE ENG. YAPARAK ALDIGIMIZ SKOR.

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

#gorduk ki grafikte. sex_male bize en cok yarıyan özellij olmus kendi atadıgımız.



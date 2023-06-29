# Bir ürünü satın aldıran nedir?

# bir ürün alırken bi siteye girip arama yaptıgımızda fiyatı satın alanların yorumları ve puanı vardır onlara bakarız.


##Social Proof:

# bir ürüne diğerlerinin yaptığı yorum puanlama vs. eğer yüksekse biz kalabalığın bu bilgeliğine (the wisdom of crowds) inanırız.

##Wisdom of Crowds:

# herkes köprüden atlıyorsa ben de atlarım!

# orn 7 degerlendırme uzerınden 5 yıldız almış diğeri ise 256 degerlendirmeden 4.6 puan almış. puan düşük olmasına rağmen 4.6yı tercih ederiz.

# ürün puan hesaplaması

# ürünlerin sıralanması

# ürün detay sayfalarındakı yorumların sıralaması

# sayfa sürec etkileşim alanların tasarımı, renk seçimi

# özellik denemeleri (yeni özellik ekledim nasıl sonuc verdi)

# olası aksiyonların ve reaksiyonların test edilmesi..

# RATING PRODUCST

# SORTING PRODUCTS

# SORTING REVIEWS

# AB TESTING

# DYNAMIC PRICING


###URUN PUANLAMA (RATING PRODUCTS)

# OLASI FAKTÖRLERI GOZ ONUNDE BULUNDURARAK AGIRLIKLI URUN PUANLAMA ISLEMI. ORN VERILEN YILDIZLARA GÖRE URUNUN PUANI.

###ORTALAMA(AVERAGE)

# AVERAGE

# TIME BASED WIEGHTED AVERAGE

# USER BASED WEIGHTED AVERAGE

# WEIGHTED RATING


import pandas as pd

import math

import scipy.stats as st

from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)

pd.set_option("display.max_rows", None)

# pd.set_option("display.width",500) thats alternative option

pd.set_option("display.expand_frame_repr", False)

pd.set_option("display.float_format", lambda x: "%.5f" % x)

##UYGULAMA: KULLANICI VE ZAMAN AGIRLIKLI KURS PUANI HESAPLAMA


df = pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/course_reviews.csv")

df.head()

# 50 saat python veri bilimi ve ml

# puan 4.8

# toplam puan 4611

# puan yüzdeleri 75 20 4 1 1

# yaklasık sayısal karsılıkları: 3458, 922, 184, 46, 6


df.shape  # 4323 tane puanlama var.

df["Rating"].value_counts()  # rating dagılımı

df["Questions Asked"].value_counts()

df.groupby("Questions Asked").agg({"Questions Asked": "count",

                                   "Rating": "mean"})

df.head()

# ortalama hesabı

df["Rating"].mean()

# boyle bir ortalama hesabında ürünlerle ilgili son zamanlardaki trendi kaçırabiliriz. memnuniyet trendi. örn ilk 3 ay çok iyi puan aldı, sonra baya düşmüş olabilir.


####Time-Based Weighted Average


df.head()

df.info()

# timestamp object. bunu timeseriese cevirmek lazım.

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df.info()

current_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (current_date - df["Timestamp"]).dt.days

df.head()

# artık kaç gün oncesinde son timestamp yapılmıs goruyoruz days degiskeniyle.


df.groupby("days").agg({"Rating": "mean"}).head()

df[df["days"] <= 30].count()  # 194 veri var.

df.loc[df["days"] <= 30, "Rating"].mean()  # son 30günkülerin ortalaması

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()  # 30-90 arası

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()  # 90-180 arası

# son zamanlarda hala yüksek ortalamaya sahip.

# her zaman dilimine bi ağırlık verip bir katsayı gibi bi şey oluşturabiliriz.

df.loc[df["days"] <= 30, "Rating"].mean() * 50 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100


# en yakin tarihe yuksek katsayi verdik cunku amac son zamanlardaki puanlamaya agirlik vermekti


# bunu bir func olarak yazalım


def time_based_weighted_average(dataframe, w1=50, w2=26, w3=24):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
 \
        dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
 \
        dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * 24 / 100


time_based_weighted_average(df)

# son zamanlar onemsız eskı tarihler daha onemli olsaydı...


time_based_weighted_average(df, 20, 30, 50)  # 3,76ya dustu puanımız.

###USER-BASED WEIGHTED AVERAGE

# HER KULLANICI AYNI AGIRLIGA MI SAHIP OLMALI? YOKSA KURSU IZLEYEN IZLEMEYENE GORE DAHA MI AGIR OLMALI?

# ORN IMBDE YENI UYE OLANLA SUREKLI OYLAYAN ADAMLA BIR DEGIL. COBANLA BENIM OY

df.head()
df.groupby("Progress").agg({"Rating": "mean"}).tail()


def user_based_weighted_average(dataframe, w1=20, w2=30, w3=50):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
 \
        dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 60), "Rating"].mean() * w2 / 100 + \
 \
        dataframe.loc[(dataframe["Progress"] > 60) & (dataframe["Progress"] <= 100), "Rating"].mean() * w3 / 100


user_based_weighted_average(df)


# simdi ise hem zaman hem kullnıcı bazlı bir ağırlıklı average alalım ve bu en geneli olsun..


def course_weighted_rating(dataframe, userw_=50, timew_=50):
    return user_based_weighted_average(dataframe) * userw_ / 100 + time_based_weighted_average(dataframe) * timew_ / 100


course_weighted_rating(df, userw_=60, timew_= 40)
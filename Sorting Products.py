# Ürün Sıralama(Sorting Products)

# bir iş ilanına başvuran adayları dusunun. bu adayların 3 tane puanı oldugunu dusunun.

# 1. not ort. 2. mulakat puan. 3.dil bilgisi gibi . neye göre secim yapacagız. yine katsayılı bir sıralama yapabilriz. mulakat yuzde 50 digerleri 25 vs.

# e-tic sıralama görecegiz.urunle alakalı bilgilere göre yapılacak puanlamanın sıralamasıyla ılgılenecegız.


##DERECELENDIRMEYE GORE SIRALAMA (SORTING BY RATING)

import pandas as pd

import math

import scipy.stats as st

from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)

pd.set_option("display.max_rows", None)

# pd.set_option("display.width",500) thats alternative option

pd.set_option("display.expand_frame_repr", False)

pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/product_sorting.csv")

df.head()

# ratinge göre sıralayalım.

df.sort_values("rating", ascending=False).head(10)

# ratinge göre sıraladık ama bazılarının commenti ya da yorum sayıları çok düşük. neye göre en iyisi diyeceğiz. işte bu noktada bazı miksler yapmamız gerekecek.


# yorum + satın alma sayısına göre sıralama:


df.sort_values("purchase_count",
               ascending=False).head()  # ama ratingini bu sefer yakalayamıyoruz.yorum sayısı da kaçıyor.

df.sort_values("commment_count",
               ascending=False).head()  # yine aralarında en iyi sonuc ama hala problemler devam ediyor.

# bu 3 faktörü aynı anda degerlendirelim. belirli bi standartta birleştirip.


##derecelendirme, satın alma , yoruma göre sıralama...


# standartlaştırma işlemi yapmamız lazım.


from sklearn.preprocessing import MinMaxScaler

# hepsini carpıp bi columnda olustursak fena fikir degil. ama ratingi çok ezmiş oluruz rating maks 5 olan bir degisken comment ise 5klara purch count 50klara cıkabiliyor.


df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[["purchase_count"]]).transform(df[["purchase_count"]])

df.head()

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[["commment_count"]]).transform(
    df[["commment_count"]])

df.head()

df["comment_count_scaled"] * 32 / 100 + df["purchase_count_scaled"] * 26 / 100 + df["rating"] * 42 / 100
#kafamıza göre bu 3 degiskene katsayılar atayıp bir skor yarattık.

# func yapalım


def weighted_total_score(dataframe, w1=32, w2=26, w3=42):
    return dataframe["comment_count_scaled"] * w1 / 100 + dataframe["purchase_count_scaled"] * w2 / 100 + dataframe[
        "rating"] * w3 / 100


df["total_score"] = weighted_total_score(df)
df.head()

# sadece veri biliminin kurslarını baz alalım.
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("total_score", ascending=False).head(10)

###Bayesian Average Rating Score

##bu ratingleri acaba daha farklı acıdan hassaslastırabilir miyiz? sadece ratinge odaklanarak bir sıralama yapabilir miyiz?

##ratingleri direkt görüyoruz ve bir de kaç tane 5 yıldız kaç tane 4 yıldız 3,2,1leri görüyoruz.şimdi onlara odaklanarak yeni bir score yaratacagız

# puan dagılımı uzerınden bır ortalama.

import math

import scipy.stats as st

def bayesian_average(n, confidence=0.95):
    """given a dataframe, returns a series of bayesian averages"""
    if sum(n)==0:
        return 0
    K = len(n)
    z = st.norm.ppf(1-(1-confidence)/2)
    N = sum(n)
    first_part= 0.0
    second_part = 0.0
    for k,n_k in enumerate(n):
        first_part += (k+1) * (n[k]+1) / (N+K)
        second_part += (k+1) * (k+1) * (n[k]+1) / (N+K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N+K+1))
    return score
df.head()
df["bar_score"]=df.apply(lambda x: bayesian_average(x[["1_point","2_point","3_point","4_point","5_point"]]),axis=1)
df.head()

#artık ratingler için cok saglam bir skor var ama hala eksiklikler var. 4-5 yıldızı olmayanlara çok yüksek skor verdi ama comment vs dusuk olmasına ragmen.

#simdi hibrit bi sıralama yapalım en kapsamlı....

##Hybrid Sorting

##bu zamana kadar ne yaptık?
#average
#time based weighted average
#user-based weighterd average
#weighted rating
#bayesian average rating score(bunu yapmadık. cok da onerilmez cunku bayes cok puan kırdıgı ıcın)
#sorting by rating
#sorting by comment count or purchase count
#sorting by rating comment purchase
#sorting by bayesian average
###SİMDİ HYBRiD: bayesian + diğer faktörler


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)
    wss_score = weighted_total_score(dataframe)

    return bar_score * bar_w/100 + wss_score * wss_w/100


df["hybrid_score"] = hybrid_sorting_score(df)


df.sort_values("hybrid_score",ascending=False).head()

#bar score yorum sayısı düşük olsa bile umut vadeden bir ortalama verebilir. o sebeple bayes+kendi yayptıgımız score miksi en doğru score diyebiliriz.


###IMDB FILM PUANLAMA VE SIRALAMA UYGULAMASI


df= pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/movies_metadata.csv")
df.head() #karısık bi dataseti. bize lazım olan 3lüyü secelim.

df = df[["title","vote_average","vote_count"]]
df.head()

df.sort_values("vote_average",ascending=False).head()

#guzel ama vote countı 1 olsa bile direkt en üste cıktı.
#vote countu belli bi sayının üzerinde olanlara bakalım.

df["vote_count"].describe([0.10,0.25,0.50,0.70,0.80,0.90,0.95]).T
#ortalaması 109muş. std 491 miş. biz 434 degerini verirsek sadece yüzde 5lik bir niş liste yaparız. şu anlık okay bi durum.


df[df["vote_count"]>430].sort_values("vote_average",ascending=False).head(10)
#vote countu da 0-10luk dilime sokalım ki bir score oluşturalım. öbür türlü 2sini carpsak vote avg çok ezilir.
from sklearn.preprocessing import MinMaxScaler
df.head()
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])

#artık vote_Avg ve vote_count_scorre 0-10luk dilimde. ikisini çarpıp bir score oluşturalım ve ona göre sıralayalım...

df["total_score"] = df["vote_count_score"] * df["vote_average"]

df.sort_values("total_score",ascending=False).head(10)
#su an güzel bir sıralama yaptık gibi. en iyi ilk 10 filmi gördük gibi.... daha ileriye gidelim....


###IMDB AGIRLIKLI DERECELENDIRME (IMDB WEIGHTED RATING)

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C) BU IMDBNIN UYGULADIGI FORMUL

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

#OZETLE COK YUKSEK OY ALSA DAHI YORUMA BAKAR. COK YORUM ALSA DAHI PUANI MIXLER. GUZELBIR HESAPLAMA

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)
df.head()
df.sort_values("total_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.00000, M, C) #deadpool icin bizimki 7.40 ken 7.08 verdi bu.

weighted_rating(8.10000, 14075.00000, M, C) #

weighted_rating(8.50000, 8358.00000, M, C)#esaretin bedeli

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)  #hepsi için yaptık..

df.head()

df.sort_values("weighted_rating", ascending=False).head(10)

#ama yetmez. bayesi de katalım işin içine...

#imdbnin kendi listesinde user quality + bayesian mixine göre. bizde user quality datası yok ama bayesiane göre bakalım.

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351]) #esaretin bedeli 1yıldızdan 10 yıldıza kadar(1yıldız veren to 10 yıldız verene)
#9.14 verdi esaretin bedeline.
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])
#godfather
df = pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/imdb_ratings.csv")
df = df.iloc[0:, 1:]
#yukarıdaki 1den 10a yıldız sayısını direkt data setinden cektik. simdi bir column olsuturalım ve bar_scoreları girelim.
df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)


#ama unutmamak lazım ki (tekrardan) biz sadece bayese baktık. örn hintli bir filme 1milyar hintli 10 verebilir. user quality index o sebeple çok önemli.
#burada ona bakamıyoruz.


# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.



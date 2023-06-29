###AB TESTING
#A BIR GRUBU, ISTATISTIGI TEMSIL ETSIN, B BASKA Bİ TANEYİ. IKISI ARASINDAKI KARSILASTIRMA FARKLILIK BULMA ISLEMI KISACA AB TESTING.
#AMA ONCE BAZI ISTATISTIKSEL KAVRAMLARA BAKALIM..

#SAMPLING(ORNEKLEME):  POPULASYON ORNEKLEMDEKI ORNEKLEM.. DAHA AZ VERI.

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal

from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",10)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

populasyon = np.random.randint(0,90,10000) #0-90 yas arası 10k insanlık bir populasyon olustrudk
populasyon.mean() #ort 44.7042


np.random.seed(115) #bunu vahit hocayla aynı random degerlere ulasabilmek icin yazdık
orneklem = np.random.choice(a=populasyon,size=100)
orneklem.mean() #44.67

np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon,size=100)
orneklem2 = np.random.choice(a=populasyon,size=100)
orneklem3 = np.random.choice(a=populasyon,size=100)
orneklem4 = np.random.choice(a=populasyon,size=100)
orneklem5 = np.random.choice(a=populasyon,size=100)
orneklem6 = np.random.choice(a=populasyon,size=100)
orneklem7 = np.random.choice(a=populasyon,size=100)
orneklem8 = np.random.choice(a=populasyon,size=100)
orneklem9 = np.random.choice(a=populasyon,size=100)
orneklem10 = np.random.choice(a=populasyon,size=100)

#10unun birden ortalamasını alalım..

(orneklem1.mean() + orneklem2.mean() +orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

#ort. 43.872


###BETİMSEL İSTATİSTİKLER(KESİFCİ VERİ ANALIZI) (Desribetive Statistics)

df= sns.load_dataset("tips")
df.describe().T

##veri setinde aykırı değerler varsa meanini almak çok sağlıklı olmayabilir. %50ye denk gelen ortanca değer daha sağlıklı.
#bir veri seti hakkında mean ve %50inci degere bakarak yorum yapılabilir. eger yakınsa degerler cok aykırı deger yok gibi düsünülebilir.
#std mean ve %50 avg degerleri genel bir bakış için önemli olabilir.


###Confidence Intervals(Güven Aralıkları)

#Anakütle parametresinin tahmini değerlerini(istatistik) kapsayabilecek iki sayıdan oluşan bir aralıktır.

df.head()

sms.DescrStatsW(df["total_bill"]).tconfint_mean() #total_bill güven aralıkları...
sms.DescrStatsW(df["tip"]).tconfint_mean() #total_bill güven aralıkları...

#titanic veri setine bakalım..

df= sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean() #total_bill güven aralıkları...
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean() #total_bill güven aralıkları...

#burada dikkat ettiysek farein güven aralıgı cok daha büyük. sebebi farein standart sapmasının yuksek olması
#formulde standart sapmanın karesi var. o sebeple bu durumlarda güven aralıkları büyür..


##Correlation(Korelasyon)
#degiskenler arasındaki ilişki, ilişkinin yönü ve siddeti hakkında bilgi veren istatiktsel yöntemdir.

df=sns.load_dataset("tips")
df["total_bill"]=df["total_bill"]-df["tip"] #tipsiz bill icin yaptık

df.plot.scatter("tip","total_bill")
plt.show(block=True) #pozitif bir korelasyon gördük. bakalım kaçmış?=


df["tip"].corr(df["total_bill"]) #0.5766ymış.

###Hipotez Testleri:
#bir inanışı bir savı test etmek için kullanılan yöntemlerdir. Bizim yapacagamız, grup karşılaştırmalarında temel amaç,
#olası farklılıkalrın şans eseri çıkmadığını göstermeye calısmaktır

#örn. mobil uygulamada arayüz degistikten sonra uygulamayı kullananların geçirdiği süre arttı mı ?
#bazı örnekleme yeni arayüz sunuldu ve 58dk ort süre gecirdiler. eski arayüzü kullanan örneklemdekiler ise 55dk. kağıt üstünde yeni arayüz daha iyi. ama öyle mi? işte bunu test etme işidir. yapacagımız şey. şans mı değil mi?
#2 aşı firmasının koruma oranı karşılaştırması. a firm: 80 b firm:84(eşit koşullarda). işte burada hipotez testi yapıp bilimsel olarak test edecegiz.


###AB TESTING (BAGIMSIZ IKI ORNEKLEM T TESTI)
#iki grup ortalaması arasında karşılaştırma yapmak için kullanılan test.
#H0: ilk durum, H1:yeni durum
#H0: X1  = X2   or  #H0: X1 <= X2   or #H1: X1 >= X2
#H1: X1 != X2       #H1: X1 >  X2       #H1: X1 < X2

#ilk duruma focus olacagız..

#hipotez kur
#varsayımları incele
#p valueya bak. yorum yap

#elle matematik kullanarak hesaplasaydık. th degeri < tt olunca H0 reddedilemez diyecektik. kodlamada p valueya bakacagız.
#p value < 0.05 = H0 RED. Yani H0ın yanlıs onerme olma durumu. (yukarıdaki senaryo için X1=X2 değil). H1e bakmak lazım...
#orn: bir uygulama 1 yıl yeni arayüz geliştirdi ve sundu. istatistikleri elimize ulastı. yeni arayüzde ort. daha yüksek her şey iyi gözüküyor ama,
#p>0.05 geldi yaptıgımız hipotezde. yani H0 reddedilemez. H0 neydi? X1=X2 olma durumu. yani yeni arayüz bi işe yaramadı eskisiyle aynı çıktı.

#bir ornekle bakalım...

#Sigara içenlerle içmeyenlerin hesap ortalamaları arasında fark var mı ??

df=sns.load_dataset("tips")
df.groupby("smoker").agg({"total_bill":"mean"}) #kagıt üzerinde fark var gibi duruyor.

#teste baslayalım. ilk olarak;

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2 (yani sigara icenlerin total_billle icmeyenlerin eşit olma durumu)
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği


###2.A. Normallik Varsayımı:
##normallik varsayımlarının hipotezi şöyle der:
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.
test_stat, pvalue = shapiro(df.loc[df["smoker"]=="Yes" ,"total_bill"])
#smoker=yes için.

print(test_stat,pvalue) #gorduk ki pval= 0.0002. pvalue < 0.05

#p val  < 0.05. yani H0 RED. Yani Normal dağılım varsayımı sağlanmamaktadır.


test_stat, pvalue = shapiro(df.loc[df["smoker"]=="No" ,"total_bill"])
#smoker=no için.

print('Test Stat = %.4f, p-value = %.4f'% (test_stat,pvalue)) #gorduk ki pval= 0.0002. pvalue < 0.05
#pval again. <0.05

#H0 RED AGAIN.

###2.B VARYANS HOMOJENLİGİ VARSAYIMI:

#hipotezlerimiz şöyle der:
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#pval 0.0452 yani <0.05. H0 RED.


############################
# 3 ve 4. Hipotezin Uygulanması
############################
#YUKARIDAKI VARSAYIMLARA GORE HIPOTEZ UYGULAYACAGIZ. VARSAYIMLARA GÖRE NAPIYORDUK?
# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

#ORNEGIMIZDE SAGLANMIYOR. NON-PARAMETRIKLE DEVAM ETMEMIZ GEREKIYOR. AMA BIZ GOSTERMEK ICIN HER 2SINI DE YAPALIM.

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
#her ikiside(varyans ve normal dagılımı varsayımları) saglanıyorsa true, norm saglanıyor, varyans saglanmıyorsa false yapacagız. false yapınca baska bir test calıstırack aslında.

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p = 0.1820
#p>0.05 çıktı yani H0 REDDEDILEMEZ.
#H0 NEYDİ? X1=X2 yani sigara içenlerle içmeyenler eşit tip bıraktı.
#onermemiz dogru degil....


############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

test_stat,pvalue=mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#pval = 0.3413
#pval >0.05 yani H0 Reddedilemez.
#H0 NEYDİ? X1=X2 yani sigara içenlerle içmeyenler eşit tip bıraktı.
#onermemiz dogru degil....


############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################
df = sns.load_dataset("titanic")

df.head()

##ISTATIKSEL OLARAK KADIN-ERKEK YAŞ ORT ARASINDA ANLAMSAL BI FARK VAR MI?

df.groupby("sex").agg({"age":"mean"}) #burada fark gözüküyor. check edelim..

#1. hipotezi kur.
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (vardır)

#2. varsayımları incele.
#2.1 normallik varsayımı
#H0: normal dağılım varsayımı sağlanmaktadır.
#H1: sağlanmamaktadır.
 #normallik varsayımı için shapiro.
test_stat,pvalue = shapiro(df.loc[df["sex"]=="female","age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#pval = 0.0071 < 0.05 yani HO RED.
test_stat,pvalue = shapiro(df.loc[df["sex"]=="male","age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#pval=0.000... < 0.05 HO RED.
#RED YEDİK YANİ H0I REDDETTIK. YANİ SAGLANAMAMAKTADIR. DİREKT NONPARAMETRİKTE İŞLEM YAPMAMIZ GEREKTİĞİNİ BİLİYORUZ.

#SON KEZ GORMEK İÇİN Bİ DAHA VARYANS HOMOJENLIGINI DE CHECK EDELIM.
#varyans homojenligi = levene

test_stat,pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                          df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#pval =0.97 >0.05 yani H0 REDDEDİLEMEZ. YANİ VARYANS HOMOJEN DAGILMIŞTIR AMA BİR ONEMİ YOK.


# Varsayımlar sağlanmadığı için nonparametrik
#nonparametrik = mannwhitneyu
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#p 0.0261 <0.05 yani H0 RED.YANİ KADIN ERKEK YAŞ ORT ARASINDA BİR FARK VARDIR.

#YAŞ ORTALAMALARI İÇİN GENELDE ÇOK AB TESTİNG YAPMAYIZ CUNKU ZATEN ACIK GIBI...
#DAHA COK BIR UYGULAMA ARAYUZUNDEN SONRA YENI ZAMAN ORT.,AŞIDAN SONRA BAGISIKLIK ORT.GİBİ BİLMEDİGİMİZ ŞEYLER İÇİN TEST YAPARIZ.


############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df= pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/diabetes.csv")
df.head()
df.groupby("Outcome").agg({"Age": "mean"}) #var gibi duruyor.

#1.hipotezi kur.
#HO: M1=M2(Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Fark yoktur.)
#H1: M1 != M2 (vardır)

#2. Varsayımları incele...
#normal dagılım.
#HO: Normal dagılım saglanmıstır.
#H1: Saglanmamıstır.
#normallik = shapiro
st_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p= 0.000 < 0.05 yani HO:RED. Diabeti olanlar için.
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p= 0.000 < 0.05 yani HO:RED. Diabeti olmayanlar için

#direkt non-parabolice gecelim.
#nonparabolic = mannwhitneyu
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p=0.000 < 0.05 yani HO:RED Yani yaş ortalamaları arasında bir fark vardır..


###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

df=pd.read_csv("DERSLER/MEASUREMENT PROBLEMS/course_reviews.csv")
df.head()

df[df["Progress"]>75]["Rating"].mean() #4.8604 #baya izleyenlerin puanı
df[df["Progress"]<25]["Rating"].mean() #4.72 az izleyenlerin oranı

test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#PVAL=0.000 <0.05, H0=RED

test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#P=0.000 < 0.05, H0=RED
#NONPARABOLIC
test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#P=0.000 < 0.05 HO:RED
#Yani puanlar arasında fark vardır...

##BU ZAMANA KADAR GRUP ORTALAMALARI (MEDIANLARINA BAKTIK) SIMDI IKI GRUP ICIN ORAN KARSILASTIRMASI YAPALIM....

######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır


#kendi elimizle yazalım..
basari_sayisi =np.array([300,250]) #1.durumun basarı sayıs, 2.durumun basarı sayusı
gozlem_sayisi = np.array([1000,1100])#1.durumun gozlem sayusı, 2.durumun basarı sayusı
#2li grup oran testinde proportions ztest yapılır...

proportions_ztest(count=basari_sayisi,nobs=gozlem_sayisi)
#p=0.00001 < 0.05 H0:RED yani iki tasarım arasında fark vardır...
#peki hangisi daha iyi???
basari_sayisi/gozlem_sayisi
#1. 0.3, 2: 0.22  yani 1. daha iyi...

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df= sns.load_dataset("titanic")
df.head()

df.loc[df["sex"]=="female","survived"].mean() #kadınların yaşama oranı #0.74
df.loc[df["sex"]=="male","survived"].mean() #kadınların yaşama oranı #0.18


#basarı oranı ve toplam sayı lazım....

female_succ_count = df.loc[df["sex"]=="female","survived"].sum() #yasayan kadın sayısı 233
male_succ_count = df.loc[df["sex"]=="male","survived"].sum() #yasayan erkek sayısı 109

df.loc[df["sex"] == "female", "survived"].shape #(314, ) cıktısı
df.loc[df["sex"] == "male", "survived"].shape #(577,) cıktısı


test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#pval = 0.000 < 0.05 H0:RED yani ortalamalar arasında bi fark vardır...


######################################################
# ANOVA (Analysis of Variance)
######################################################
#aslında cok uzun bir konu. biz burada sadece 2den cok grup ort. arasında bi fark var mı yok muyla ilgilenecegiz. thats all..

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

#günlere göre total bill ort. farklı mıdır?

df.groupby("day")["total_bill"].mean()
# Thur   17.68274 Fri    17.15158 Sat    20.44138 Sun    21.41000

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: sağlanmamaktadır.

for x in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"]== x, "total_bill"])[1]
    print(x, 'p-value: %.4f' % pvalue)

#hepsi 0.05ten kucuk yani HO:RED. 4UNE TOPLU BAKINCA RED GELDI.

#Varyansa bakalım...
test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#varyans >0.05 H0:KABUL...

# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

#PVAL = 0.042 <0.05 YANI HO:RED

#AMA BIZE NON PARAMETRIK LAZIM CUNKU NORMAL DAGILIM RED YEDI. NON PARAMETRIK KRUSKA

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

#PVAL = 0.015 < 0.05 YANI RED. H0 RED TOPLUDA
#2LI ILISKILERE BAKALIM........

from statsmodels.stats.multicomp import MultiComparison

comparison = MultiComparison(df["total_bill"],df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

#2liye bakınca p-adj hepsi 0.05ten buyuk. yani H0:Kabul.

#burada ya 0.05 alpha degeriyle oynarız. ya da ikilide h0 kabul oldugu icin biz de H0 degerini kabul ederiz...


###QUIZZZZZZZZ
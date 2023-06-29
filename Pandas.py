##############################Pandas
# veri manipulasyonu, veri analizi.

###########Pandas Serileri...
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])  # pandas serisi olusturma
type(s)
s.index
s.dtype
s.size
s.ndim  # kac boyutlu oldugunu goruyoruz seri 1 boyut
s.values  # içindeki değerler.
type(s.values)
s.head(3)  # ilk 3 value
s.tail(3)  # sondan 3  value

############Veri Okuma..

# csv dosyası okuma..

df = pd.read_csv("winemag-data_first150k.csv")

df.head()
# pd yazıp ctrl tık yapınca acılan sayfada read yazınca hangi dosya türünü nasıl okuyacagını gorebılırsın.

#############Veriye Hızlı bir bakış..

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T #DAHA okunaklı oluyor .T ile
df.isnull().values.any() #true cıktı demekki varmiş.
df.isnull().sum() # columnslardaki nulları gösteriyor.
df["sex"].value_counts() # kategorik degiskenin sınıflarını ve kaç tane oldugunu gösterdi..

#############Pandas Seçim İşlemeleri..(Selection in Pandas)

#veri analizi, manip. diyince akla seçim işlemleri gelir.

df.head()
df.index
df[0:13] #0dan 13e kadar git.
df.drop(0,axis=0).head() #0 = 0.index. axis=0 demek satırlardan sileceğim demek.
df_deletedindex= [0,1,3,4] #bunları toplu silmek istiyorum.
df.drop(df_deletedindex,axis=0).head()
##bunlar kalıcı degildi. tekrrar df atayarak kalıcı yapabiliriz. ya da inplace yaparsan.
df.drop(df_deletedindex,axis=0, inplace=True).head() # artık kalıcı oldu silinenler.

#########Değişkeni Indexe Çevirmek

#SIK SIK KARSILACAGIZ..

df["age"].head()
df.age.head()
df.index = df["age"] #artık df in indexi agedeki valuelar oldu.
#age artık index oldu. age columnuna gerek kalmadı. silelim.
df.drop("age", axis=1, inplace=True).head() #bu sefer column sildik, axis 1.
df.head()
#age artık bir index. tekrar yerine columna koyalım.

####### Indexi Degiskene Cevirmek...

df["age"] = df.index #tekrar ekledik.
df["age"].head()

##ya da tek kod.
df = df.reset_index() #indexi sıfırladı.eski index valuelarını column yaptı.
df.head()
df["age"]


##########Degiskenler Üzerinde İşlemler..

"age" in df #varsa true döndürdü.

#bir değişken seçerken sonucu series ya da df olarak alabilirsiniz.

type(df["age"]) #bu bir pandas series i
type(df[["age"]]) # bu ise bir pandas dataframei.
## iki parantezle df oldu.

cols=["age","adult_male","alive"]
df[cols]  #tek tek yazmak yerine önce yukarda liste yaptık.age

df["age2"] = df["age"]**2
df.head()
df["age3"] = df["age"]**2 / df["age"]*4

del_cols=["age2","age3"]

#otomatik olarak istediğimiz şeyi silmek...

df.loc[:, ~df.columns.str.contains("age")].head()
##tilda ~ degildir demek. age olmayanları sırala dedik. " : " derken hepsi demek.
#df.columns.str.contains("age") bunu aratınca içinde age olan columnsları arıyor. 3 tane buldu. age age2 age3



##################Loc & Iloc

##dflerde secim islemi icin kullanılır.
#loc: label based
#iloc: intiger based

df.head()
df.iloc[0:3] #indexi 3e kadar olanlar.
df.iloc[0,1] #0=satır 1=sütun.

df.loc[0:3]  #label based oldugu için 3ü de aldı.
#loc: isimlendirmenin kendisini seçiyor.

#isimlerle seçim yapmak istersek label based. iloc hata verecek.
df.iloc[0:3,"age"] #hata verdi cunku index bekliyor.
df.loc[0:3,"age"]

col_names = ["age","alive"]

df.loc[0:4,col_names]

############### Koşullu Seçim (Conditional Selection)
#yaşı 50den büyük olanlar..
df[df["age"]>50].head()
#yaşı 50den büyük kaç kişi var?
df[df["age"]>50].count() #böyle yapınca her column için sonuç verdi. spesifik olarak hangisini istiyorsak...
df[df["age"]>50]["age"].count() #64 kişi.

##yaşı 50den büyük olanların class ve age sınıfını yazdır.

df.loc[df["age"] > 50 , ["class","age"]].head()

#bir de erkek ekleyelim. #hata verecek. her condition parantez icine alınmalı.
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["class","age"]].head()

#3.kosulu ekleyelim. embark_townı soton olanlar.
df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & (df["embark_town"] == "Southampton"),
       ["class","age","embark_town"]].head()
#embarkı soton ya da Cherbourg olan..
df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Southampton") | (df["embark_town"] == "Cherbourg")),
       ["class","age","embark_town"]]
df_new["embark_town"].value_counts()


############Toplulaştırma & Gruplaştırma(Aggregration and Grouping)
#toplulaştırma bize özet istatistikler verir.
#count()
#first()
#last()
#mean()
#median()
#min()
#max()
#std()
#var()
#sum()
#pivot table.
##gruplaştırma ise toplulaştırmayla  birlikte kullanılır.

df.head()

#kadınların ve erkeklerin yaş ortalaması?
df["age"].mean() #bu total yaş ortalaması..
df.groupby("sex")["age"].mean() #ilgili değişkene (sex) aggriagtion yaptık.

#toplamını da al.
df.groupby("sex").agg({"age": ["mean","sum"]})
#agg fonksiyonunda ilk olarak ilgili değişken, sonra istenilen fonksyionlar yazıldı,
#liste yapmak mantıklı ki birden fazla değeri girebildik.


#embark towna göre de frekans bilgisi gir cinsiyete göre.
df.groupby("sex").agg({"age": ["mean","sum"],
                       "embark_town": "count"})
#bana aslında embarkın değil cinsiyetin frekansını verdi.

#mesela hayatta kalma oranına ulasalım.

df.groupby("sex").agg({"age": ["mean","sum"],
                       "survived": "mean"})

#sadece cinsiyete göre değil, liman biniş yerine göre de kıralım.

df.groupby(["sex","embark_town"]).agg({"age": ["mean","sum"],
                       "survived": "mean"})

#class bilgisine göre de bakalım.
df.groupby(["sex","embark_town","class"]).agg({"age": ["mean","sum"],
                       "survived": "mean"})

#bu veri setine dair çok fazla bilgiye yukarıdaki kodla ulaşabiliriz.
#her şey çok güzel ama frekans bilgisi yok. tam exact yorum yapamıyoruz. ekleyelim.

df.groupby(["sex","embark_town","class"]).agg({
    "age": ["mean","sum"],
    "survived": "mean",
     "sex": "count"}) #sex:count ile kaç tane oldugunu sayıdrıdık.


#############Pivot Table..
#cinsiyet ve classa göre yaşama oranını pivot table yapalım.
df.pivot_table("survived","sex","class")
#1. kesisimde neyi görmek istiyorsan onu yaz.(ortalama degeri verecek. pivotun ön tanımlı degeri mean dir.)
#2. satırda neyi görmek istiyorsun
#3. sutunda ne görmek istiyorsun.

df.pivot_table("survived","sex","class",aggfunc="std")
#std sapmasını verdi.

df.pivot_table("survived","sex",["embarked","class"])

#hem cinsiyet,lokasyon ve yaşlara göre kırılım.
#veri setindeki yaş intiger. bunu gruba sokamayız. kategorik yapalım. cut ve qcut func. ile

#elimdeki sayısal degiskeni biliyorum = cut
#elimdeki sayısal değişkeni bilmiyorun= qcut. (çeyreklik değere böler.)

df["new_age"] = pd.cut(df["age"],[0,10,18,25,40,90]) #neyi(1.değer), nereden(2.input) böleceğimi ver.
#0-10 bi kategori, 10-18, 18-25, 25-40, 40-90 bi kategori.

#hem cinsiyet,lokasyon ve yaşlara göre kırılım.

df.pivot_table("survived","sex","new_age")
#bir boyut daha. yolculuk sınıfı.
df.pivot_table("survived","sex",["new_age","class"])
#son pivot tableda cinsiyete, yaşa, sınıfa göre yaşama oranına baktık.

pd.set_option("display.width",500) #daha güzel okunabilir oldu.
df.pivot_table("survived","sex",["new_age","class"])
df.drop("new_age",inplace=True,axis=1)

##############APPLY & LAMBDA

#APPLY= satır ve sütünlarda fonksiyon calıstırma işlemi.
#LAMBDA= FUNC. TANIMLAMA SEKLİ FAKAT KULLAN-AT.

##örnegin 3 tane yas columnu olsun ve bunları 10 a bölen bir fonksiyon.
#fonksiyonsuz, tek tek yazacağız böyle:
df["age_2"]=df["age"]*2
df["age_3"]=df["age"]*3
df["age"]/10
df["age_2"]/10
(df["age_3"]/10).head() #parantezsiz hali bir liste oldugu için df olmadıgı için hata verir.
#old school function yazmaca:
for i in df.columns:
    if "age" in i:
        #print(i) #sutunlarda gezdi sutun isimlerini yazdıracak.
        print((df[i]/10).head()) #kaydetmedik ama . kaydetmek için:
        df[i]=df[i]/10 #simdi kaydettik.

#lambda ve apply ile yapalım.

df[["age","age_2","age_3"]].apply(lambda x:x/10).head()
#ilgili columnlar seçildi. applyla lambda funcini(10a böl) uygulandı.

#eger hangi column bilmiyorsak...
df.loc[:,df.columns.str.contains("age")].apply(lambda x:x/10).head() #hepsinde gez age olanları al.

df.loc[:,df.columns.str.contains("age")].apply(lambda x:(x-x.mean())/x.std()).head() #hepsinde gez age olanları al.

##eğer lambda cok mu zorladı... önce functionı yaz sonra onu apply et.

def standart_scaler(i):
    return (i-i.mean())/i.std()
df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head() #hepsinde gez age olanları al.

#ama bunları kaydetmedik... kaydetmek için..
df.loc[:,df.columns.str.contains("age")] = df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head() #hepsinde gez age olanları al.

##########Birleştirme İşlemleri(Join)
#concat(daha cok kullanılan, seri olan) ve merge kullanacagız.
import numpy as np
import pandas as pd
m=np.random.randint(1,30,size=(5,3)) #1le 30 arası 5x3 bir numpy array oluştur
df1=pd.DataFrame(m,columns=["var1","var2","var3"]) #df oluştur. veriyi m den al columnlara var1,2,3 adı ver.
df2= df1+99

##df1le df2yi birleştir...(liste içinde)
pd.concat([df1,df2]) #index 0 dan basladı. 4 e geldi sorna tekrar 0. index bilgilerini tuttu cunku.
##ignore index ypamak lazım.
pd.concat([df1,df2],ignore_index=True)
##sutun bazında da yapabiliriz..
pd.concat([df1,df2],axis=1)


####merge ile birleştirme..

df1 = pd.DataFrame({"employees": ["mark","john","andrew","dennis"],
                    "start_date": [2010,2011,2020,2002]})
df2 = pd.DataFrame({"employees": ["mark","john","andrew","dennis"],
                    "department": ["it","elec","hr","ch"]})

pd.merge(df1,df2) #oto employees e göre birleştirdi. eger kendin secmek istersen.
df3= pd.merge(df1,df2,on="employees")
df4=  pd.DataFrame({"manager": ["pep","jurgen","eddie","miguel"],
                    "department": ["it","elec","hr","ch"]})
###Her calısanın mudur bilgisi...
pd.merge(df3,df4, on= "department")


##########################  QUIZZZZ

series= pd.Series([1,2,3])
series**2

df.head()
df_4=pd.DataFrame(data=np.random.randint(1,10,size=(2,3)),
                  columns=["var1","var2","var3"])
df_4
df_4[(df_4.var1 <=5)][["var2","var3"]] # var1in 5ten kucuk oldugu yerdeki var2,var3 degerleri..

df.index.values
df.index
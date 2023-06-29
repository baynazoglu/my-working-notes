###########ADVANCED FUNCTIONAL EDA (GELISMIS FONKSIYONEL K V A )
###Amaci: Elimize kücük ya da büyük bir veri geldiginde,
###fonksiyonel olarak işleyebilme işlemidir. mantık yürütebilme,analiz edebilme.

#1.Genel Resim
#2. Kategorik Degisken Analizi(Analysis of Categorical Variables)
#3 Sayısal Degisken Analizi (Numerical Variables)
#4.Hedef Değişken Analizi(Target Variables)
#5.Korelasyon Analizi(Correlation)

#Genel Resim

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")

## veriye bir genel bakış...
df.head()
df.tail()
df.shape #satır sütün bilgisi
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

#örneğin bunları her df için tek tek yazmak yerine bir function yaratalım.

def check_df(dataframe):
    print("Shape")
    print(dataframe.shape)
    print("Types")
    print(dataframe.dtypes)
    print("Head and Tail")
    print(dataframe.head())
    print(dataframe.tail())
    print("Null Values")
    print(dataframe.isnull().sum())
    print("Math Values")
    print(dataframe.describe().T)

check_df(df)
#artık amele gibi tek tek yazmayıp fonksiyonla bakabiliriz.



############Kategorik Değişken Analizi...
#oldukça önemli...
##programatik,fonksiyonel şekilde değişkenleri analiz edeceğiz.

df["embark_town"].value_counts() #hangi towndan kaç taneyi
df["sex"].unique() #unique values. male female.
df["sex"].nunique() #number of unique. 2
df["class"].nunique()

df.info()
#simdi burada hangileri kategorik degisken?
#cateogry,object ve booleanlar kategorik degisken ya baska?
#mesela survived int gözüküyor ama içeriği 1veya0. aslında boolean gibi. yani kategorik.
#pclass içeriği 0,1,3. yani kategorik değişken.
#kategorik değişken: mesela ülkeler, mesela cinsiyet vs.

#bool,cat,object olanları seç.
cat_cols = [i for i in df.columns if str(df[i].dtypes) in ["category","object","bool"] ]
cat_cols

#df["sex"].dtypes #sonuc "O" cıkacak
#str(df["sex"].dtypes) #sonuc object geldi.
df.info()
df.head()
#simdi ise sinsileri (int gözüküp categoric olanları bulma sırası...)
#mesela tipi int,float olup sınıf sayısı belli bi sayıdan küçük olanları bulalım.
num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64 ","float64"]]
num_but_cat # su an sadece intleri ve floatlari çektik..
num_but_cat = [col for col in df.columns if df[col].nunique() < 5 and df[col].dtypes in ["int64", "float64"]]
num_but_cat  #simdi 5ten kücük sınıfı olan ve int ya da float tipli olanları yazdırdık.


#str(df["survived"].dtypes)
#df.info()
#(df[].dtypes) in ["int","float"]

#mesela kategorik oldugu halde kategorik olmayanları ayıklayalım. ornegin isim adlı column olsa 1000 kişi varsa 1000 farklı data. yani hiçbir bilgi içermiyor aslında.
#demekki nunique si cok olanları da ayıklayalım... cok sınıfı olan bu tiplere  cardinal denir.

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object","bool"]]
cat_but_car # zaten yok. ama oladabilir.

#bizim istedigimiz her columnı tek çatı altında toplayalım.

cat_cols= cat_cols + num_but_cat #   cat_but_car ı cat_colsdan cıkarmamız lazımdı ama boş liste olunca cıkarmadık.
#cat_cols = [col for col in cat_cols if col not in cat_but_car] diye yazdıracaktık.
df[cat_cols]
df[cat_cols].nunique() #gördük ki en fazla 7 degiskeni olan columnlar alınmış..
[col for col in df.columns if col not in cat_cols] #ayıkladığımız columnları  check ettik.

#mesela simdi bir func yazalım. bize kaç tane veri olduğunu ve columndaki yüzdeliğini versin.
df["survived"].value_counts() #kaç veri oldugu
100* df["survived"].value_counts()/ len(df["survived"]) #yüzde kaça denk geldiği.

def cat_summary(dataframe, col_name):
    print("Kaç değer var?")
    print(dataframe[col_name].value_counts())  # kaç veri oldugu
    print("Sınıfın Yüzde Kaçı?")
    print(100 * dataframe[col_name].value_counts() / len(df[col_name]))  # yüzde kaça denk geldiği.

cat_summary(df,"survived")

##daha güzel yapalım. çıktıyı dataframe yapalım...
def cat_summary_df(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        " Yüzde Kaçı?": 100 * dataframe[col_name].value_counts() / len(df[col_name])}))
    print("------------------------------------------------")


cat_summary_df(df,"sex")


#e milyonlarca columns oldugunu varsayarsak...

for col in cat_cols:
    cat_summary_df(df, col)

# su an harika bir özet çıkardık veriyle ilgili.. yüksek seviyede özet aldık.

#cat summaryi biraz daha geliştirelim mesela... grafik ekleyelim mesela.
def cat_summary_df(dataframe, col_name,plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        " Yüzde Kaçı?": 100 * dataframe[col_name].value_counts() / len(df[col_name])}))
    print("------------------------------------------------")

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)


cat_summary_df(df,"sex",plot=True)

#bunu loopa sokup hepsine bakalım.

for col in cat_cols:
    cat_summary_df(df,col,plot=True)

#adult_maleda hata verdi. cünkü adult male bir boolean. önce onu atlayıp devam etmeye bakalım sonra onu çevirelim ve loopa sokalım..


for col in cat_cols:
    if df[col].dtypes == "bool":
        print("--")
    else:
        cat_summary_df(df, col, plot=True)
##booleanları atladık.

df["adult_male"].astype(int) #1 ve 0 lara dönüştürdü....

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary_df(df, col, plot=True)
    else:
        cat_summary_df(df, col, plot=True)

#ve bu loopta hatasız hepsini yazdırdık...

#bu döngüyü de fonksiyonun içine yazsak ve tek bir fonks. cagırınca hepsini yapsak.CEVAP HAYIR. DO ONE THING PRENSIBI. LOOPU YAZMAK YAYGIN KULLANIMDIR. AMA GÖSTERELİM..


def cat_summary_df(dataframe, col_name,plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = df[col_name].astype(int)
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        " Yüzde Kaçı?": 100 * dataframe[col_name].value_counts() / len(df[col_name])}))
    print("------------------------------------------------")

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            " Yüzde Kaçı?": 100 * dataframe[col_name].value_counts() / len(df[col_name])}))
        print("------------------------------------------------")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary_df(df,"adult_male",plot=True)

#bu sekilde daha detaylı hale getirebiliriz ama functionu cok karmasık yaptı... gerek yok. basic bir func ve bir loopla işi çözmek adaptır.



############Sayısal Değişken Analizi
df["age"].nunique()
df[["age","fare"]].describe().T
#ben bunların int oldugunu biliyorum. hiç bilemdigim bi dften nasıl secerım?
num_cols= [col for col in df.columns if df[col].dtypes in ["int64","int32","float64","float32"]] #sayısal degiskenleri sectik. ama bazıları categorik olabilir..
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64","int32","float64","float32"]  ] #num gibi olup cat olanları sectık
num_cols = [col for col in num_cols if col not in num_but_cat] #ayıkladık. num but cat olanları num cols dan sildik....

#ya da yukarıda zaten kategorikte kaydetmiştik

num_cols = [col for col in num_cols if col not in cat_cols] #cat_cols zaten aradıgımız olmaması gereken degerlerdi. tekrar num but cat tanıtmamıza gerek yok.

#peki her df te boyle hem kategorik hem sayısal için tek tek bunları mı yazacagız? hayır. bir func tanımlarız ve artık hep o funcla işlem yaparız...

#tek tek yaptıgımız işlemleri bi yerde toplayayım...
#numerical icin..
num_cols= [col for col in df.columns if df[col].dtypes in ["int64","int32","float64","float32"]] #sayısal degiskenleri sectik. ama bazıları categorik olabilir..
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64","int32","float64","float32"]  ] #num gibi olup cat olanları sectık
num_cols = [col for col in num_cols if col not in num_but_cat] #ayıkladık. num but cat olanları num cols dan sildik....
#kategorik icin
cat_cols = [i for i in df.columns if str(df[i].dtypes) in ["category","object","bool"] ]
num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64 ","float64"]]
num_but_cat # su an sadece intleri ve floatlari çektik..
num_but_cat = [col for col in df.columns if df[col].nunique() < 5 and df[col].dtypes in ["int64", "float64"]]
num_but_cat  #simdi 5ten kücük sınıfı olan ve int ya da float tipli olanları yazdırdık.
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object","bool"]]
cat_but_car # zaten yok. ama oladabilir.
cat_cols= cat_cols + num_but_cat #   cat_but_car ı cat_colsdan cıkarmamız lazımdı ama boş liste olunca cıkarmadık.
#cat_cols = [col for col in cat_cols if col not in cat_but_car] diye yazdıracaktık.
df[cat_cols]


def num_summary(dataframe, numerical_col):
    quantiles=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df,"age")

for col in num_cols:
    print(num_summary(df,col))

#kac tane num cols varsa hepsini loopa soktuk. func a bi de plot ekleyelim.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot: #if plot: eger plot true ise demek
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df,"age",plot=True)

#loopa alalım...
for col in num_cols:
    print(num_summary(df,col,plot=True))

########Değişkenlerin Yakalanması ve işlemlerin Genelleştirilmesi....  Capturing Variables and Generalizing Operations

##bir func yazalım ve bize direkt num_cols, cat_cols ve num_but_cat ve cat_but_car columnlarını versin biz de tek tek amelasyon yapmayalım..

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64","float32","int32"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object", "bool"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64",]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # bir kaç raporlama ekleyelim ki fonks güzel gozuksun.
    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
    #bir docstring yazalım.. #cat_th= int olup unique value 10dan kucuk olma işi. #car_th= cat olup unique 20den fazla olma işi. bu sayıları biz yazdık
     """
     Veri setindeki num_cols(numerik), cat_cols(kategorik) ve cat_but_car(kategorik ama kardinal) columnlarını verecek fonksiyon..
     Parameters
     ----------
     dataframe : dataframe #buraya typeı yazılıyor.
        verinin alınacağı ana yer.
     cat_th : int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
     car_th: int,float
       kategorik fakat kardinal olan değişkenler için eşik değeri
     Returns
     -------
     cat_cols=List
     kategorik degiskenler
     num_cols=List
     Numerik Degiskenler
     cat_but_card=List
     Cardinal Degiskenler
     Notes
     -------
      cat_cols + num_cols + cat_but_card = toplam columnlar.
      num_but_cat zaten cat_cols içersinde..
      Return olan 3 liste toplam liste sayısına eşittir.
     """

grab_col_names(df)

#bunları ekleyelim.

cat_cols,num_cols,cat_but_car = grab_col_names(df)

#direkt funcla ekleedik.
###Simdi Bütün Ögrendiklerimizi bu bölüme kadar bi getirelim...

#cat_summary
def cat_summary_df(dataframe, col_name,plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        " Yüzde Kaçı?": 100 * dataframe[col_name].value_counts() / len(df[col_name])}))
    print("------------------------------------------------")

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    cat_summary_df(df,col,plot=True)

#num_summary
def num_summary(dataframe, numerical_col, plot=False):
    quantiles=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot: #if plot: eger plot true ise demek
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,plot=True)


###### Hedef Değişken Analizi

df.info()

#her veri setinde bir hedef degisken vardır. titanic setinde hayatta kalma "survived" degiskenidir.
#mesela bir şirketin verisetinde hedef degisken müsterinin terk edip etmemesidir.
#simdi "survived" degiskenine göre kategorik ve numerikal degisken analizi yapacagız.

df["survived"].value_counts()
cat_summary_df(df,"survived")

##bagımlı degisken (survived)a göre diger degiskenlerin analizi yapıp neden survived=1 olmuş? insanlar neden ölmemiş, sebebini bulabiliriz.


##################HEDEF DEGISKENIN KATEGORIK DEGISKENLERLE ANALIZI....

#mesela cinsiyetle yaşama arasındaki baglantıya bakmak isteyebiliriz.
df.groupby("sex")["survived"].mean()
df.groupby("sex").agg({"survived":"mean"})  # bu da diger türlü hali..
#ee buna da bi func yazalım tek tek yazmaktansa..

#def target_summary_with_cat(dataframe, target, categorical_col):
   #print(pd.DataFrame({"Target Mean":dataframe.groupby(categorical_col).agg({target:"mean"})}))

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target Mean":dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df,"survived","sex")

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)

#bu for dongusuyle tüm kategorik degiskenlerin analizini yaptık.


##################HEDEF DEGISKENIN NUMERIK DEGISKENLERLE ANALIZI....
df.head()
df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age":"mean"})  # bu da diger türlü hali..

def target_summary_with_num(dataframe,target,num_col):
    print(dataframe.groupby(target).agg({num_col: "mean"}))
target_summary_with_num(df,"survived","age")
for  col in num_cols:
    target_summary_with_num(df,"survived",col)

#burada da num olarak tüm degerleri bagımlı degiskenle işleme soktuk...


############Korelasyon Analizi (Analysis of Correlation)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df= pd.read_csv("DERSLER/datasets/breast_cancer.csv")
df.head()
#dfin ilk ve son columnu gereksiz. ilk column full id. son column full nan. bunları silelim.
df = df.iloc[:,1:-1]
num_cols= [col for col in df.columns if df[col].dtype in ["int64","float64"] ]
#len(df.columns)
#len(num_cols) #30 tane num cols yakaladık.

corr= df[num_cols].corr()
corr  #df[num_cols]un correlationunu aldı..
sns.set(rc={"figure.figsize":(12,12)})
sns.heatmap(corr,cmap="RdBu")
plt.show(block=True)
#grafigini cizdirdik. 1ve-1e yakın olanlar mavi ya da kırmızıya dogru yaklastı.

########Yüksek Korelasyonlu Değişkenlerin Silinmesi....

#Bu must bir işlem değil. fakat bazen yüzlerce degiskenden birbirine cok benzer(yüksek korelasyonlu),
#degiskenlerden kurtulmak isteyebiliriz. işte onu nasıl yapacagımız görecegiz.

cor_matrix= df.corr().abs()
#yukarıda absolute degere aldık. - ve + nın su an onemı yok onemlı olan ne kadar benzerler?

#aynı degerler tekrarlanmıs cunku 2 kere yazılmıs. yani sutundaki x satırda da oldugu icin ikisine 1 demişler. yani triangleın upperiyle lowerı aslında aynı degerler.

upper_triange_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
#yani birlerden olusan bir upper triangle matris oluşturup onu booleana çevirdik.
drop_list = [col for col in upper_triange_matrix.columns if any(upper_triange_matrix[col]>0.9)]
#bir drop list olusturduk matristeki degerlerden 0.9dan büyük olan varsa drop liste ekle. silmek istedigimiz değerler.

cor_matrix[drop_list] #yüksek korelasyonlu olanları secti...
df.head()

#bunları silelim.
df.drop(drop_list,axis=1)
#30 columns 21 columns a indi.

###bu işlemleri fonksiyonlaştıralım.

def high_correlated_cols(dataframe,plot=False,corr_th=0.9):
    corr= dataframe.corr()
    cor_matrix=corr.abs()
    upper_triange_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triange_matrix.columns if any(upper_triange_matrix[col]>0.9)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list


high_correlated_cols(df)  #droplisti getirdi...
drop_list = high_correlated_cols(df)
df.drop(drop_list,axis=1)
high_correlated_cols((df.drop(drop_list,axis=1),plot=True))
#bu grafikte korelasyonlu yogunluklar kırmızı ve mavi olmayacak...


#QUIZZZZZZZZZZZZZZZZ

df['diagnosis'].value_counts()

df['diagnosis'].describe([0.25,0.50,0.75])

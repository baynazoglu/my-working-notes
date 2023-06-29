
################################ CRM(Customer Relationship Management)
import pandas as pd

##Müşteri yaşam döngüsü optimizasyonu (customer lifecycle/journey/funnel): müşterilerle kurulan etkileşimi,iletişimi çeşitli görselleştirme yöntemidir. kpi ile takip etmedir.

# bir websitede ürün satıyoruz. bir kullanıcın siteyi ziyaret etmesi ilk adım, uye olması 2.adım, 3.adımsa satın almasıdır.

##İletişim:(dil,renk,kampanyalar) yeni nesile hitaben tiktok kullanmak. bankaların nostaljik vurgu yapması.

##Müşteri edinme cabaları:

##Müşteri elde tutma calısmaları.(customer churn-musterı terk)

##cross sell- up sell satıslar. cross sell: hamburger aldıysa patates satalım. up sell: kucuk boy kolayı buyuk boya cevirsin.

##müşteri segmentasyonu çalışmaları: musterılerı alt segmentlere ayırma. 1k musterıden degerlı olanlara farklı bı yaklasım dıgerlerıne farklı gıbı.

# daha faydalıya daha fazla odak..


###################CRM ANALITIGI

# veriye dayalı bi şekilde crm yapmak.


##Temel Performans Göstergeleri (key performance indicators) KPIs

# kpi bir sirkette ilk duyacagımız ve en cok kullanılan terimlerdendir. bazı kpi ornekleri:

# şirket,departman ya da calısanların performansını degerlendırmek ıcın kullanılan matematiksel hesaplama göstergeleridir.

# orn: customer acqusition rate (musterı kazanma oranı):belirli zamanda kazanılan musterı oranı..

# customer retention rate (musterı elde tutma oranı): musterı appi indirdi. ilk3 gün kullandı sonra sıldı. işte bu orana denir.

# customer churn rate (musterı terk oranı):

# Conversion Rate: (Donusum Oranı) ilanı 10k kisi gördü 1 kisi gördü. 1/10k dır oran.

# Growth Rate: (Büyüme oranı)

# Diversity Rate: kadın/erkek oranı vs.


################KOHORT (Cohort) Analizi

##Cohort: Ortak ozellıklere sahıp bır grup ınsan.

##churn rate bir ortak özelliktir. zamana göre churn oranlarının incelenmesi. ya da retention musterı tutma oranı.


########RFM ILE MUSTERI SEGMENTASYONU (CUSTOMER SEGMENTATION WITH RFM) Recency frequency monetary..


# rfm analizi musteri segmentasyonu icin kullanılan bi teknik. musterilerin satın alma alıskanlıkları üzerinden gruplara ayırılması ve bu gruplar özelinde stratejiler geliştirebilmesini sağlar.

# crm calısmaları icin bircok baslıkta veriye dayalı aksiyon alma imkanı sağlar.


##RFM METRIKLERI..

# bu metriklere göre bazı skorlar olusturacagız.

# recency (yenilik). müsterinin bizden en son alısveris yapma ratei. 1 recency 10 recencyden iyidir. 1 gün önce alısveris yapmıstır.

# frequency sıklık: toplam islem sayısı. mehmet 10 kez alısveris yapmıs ayshe 40.

# monetary(parasal deger): kac para bıraktıgı:


df = pd.DataFrame({'Musteri Adi': ["Musteri 1", "Musteri2", "Musteri 3"],

                   'R': [80, 7, 1],

                   'F': [250, 579, 120],

                   "M": [5200, 2300, 3000]})

df

# recencyde hangi deger daha iyidir? musteri 1.

# f için hangisi? musteri 2

# m icin hangisi en iyi? musteri 1.


# RFM METRIKLERINI RFM SKORLARINA CEVIRMEK LAZIM KI KIYASLAMA YAPABILELIM. YANI HEPSINI AYNI CINSE CEVIRMEK. STANDARTIZASYON YAPMA ISI.


# 1LE 5 ARASINA ALMAK MESELA.

df_standartization = pd.DataFrame({'Musteri Adi': ["Musteri 1", "Musteri2", "Musteri 3"],

                                   'R': [1, 4, 5],

                                   'F': [4, 5, 1],

                                   "M": [5, 4, 3]})

df_standartization

# artık kıyaslama daha kolay.. recencyde en dusugu 5 yaptık..

# bunları biraraya getirerek RFM skoru oluşturulur..


df_RFM = pd.DataFrame({'Musteri Adi': ["Musteri 1", "Musteri2", "Musteri 3"],

                       'R': [1, 4, 5],

                       'F': [4, 5, 1],

                       "M": [5, 4, 3],

                       "RFM": [145, 453, 513]})

df_RFM

# yanyana getirip skorlar olusturduk. ama RFM de 111-555 arasi birsuru segment olustu. onu da guncelleyelim.


##RFM SKORLARIYLA SEGMENTASYON

# BIR FOTODAKI GRAFIK RF ORANINI ACIKLIYOR. MONETARY COK ONEMLI OLMADIGI ICIN R VE F E BAKILDI. 5V5 = CHAMP

# sımdı bu fotodakı degerlerı en sona ekleyelım

df_RFMSEGMENTS = pd.DataFrame({'Musteri Adi': ["Musteri 1", "Musteri2", "Musteri 3"],

                               'R': [1, 4, 5],

                               'F': [4, 5, 1],

                               "M": [5, 4, 3],

                               "RFM": [145, 453, 513],

                               "SEGMENT": ["At Risk", "Loyal Customer", "New Customer"]})

df_RFMSEGMENTS

# segmentteki isimler RF e göre M cok da onemlı degıl.


######RFM ILE MUSTERI ANALIZI

# İş Problemi(business problem)

# Veriyi anlama (data understanding)

# Veri Hazırlama (data preparation)

# rfm metriklerinin hesaplanması

# rfm skorlarının hesaplanması

# rfm segmentlerinin oluşturulması

# tüm sürecin fonksiyonlaştırılması


#########1.iş problemi(business problem)

# bir e-tic sirketinin musterileri segmente ayırıp bu segmentlere göre pazarlama stratejisi olusturma

# sırket genelde kurumsala satıs yapan bır sirket. ingiliz şirketin 2009-2011 arası satıslarını gösterir.


# degiskenler

# invoiceNo: fatura numarası. her ürün icin unique. C ile baslıyorsa iptal edilmiş

# stockcode: ürünün kodu. her ürün icin unique.

# descrpition: ürünün ismi

# quantity: ürün adedi. faturalardaki ürünlerden kaçar tane satıldıgı ifade eder.

# invoicedate: fatura tarihi ve zamanı

# unitprice: ürün fiyatı sterlin cinsinden.

# customerid: essiz müsteri numarası.

# country: hangi ülkeden.


# 2.veriyi anlama.

import datetime as dt

import pandas as pd

pd.set_option("display.max_columns", None)

# pd.set_option("display.max_rows",None)

pd.set_option("display.float_format", lambda x: "%.3f" % x)  # virgulden sonra kac tane gormek ıstıyorsan

df_ = pd.read_csv("DERSLER/CRM/online_retail_II.csv")
df = df_.copy()
df.head()
df["total_price"] = df["Price"] * df["Quantity"]
df.groupby("Invoice").agg({"total_price": "sum"}).sort_values(by="total_price",ascending=False,axis=0)

df.shape
df.isnull().sum()
#CUST ID 250K EKSIK DEGER VAR. DIREKT SILMEK MANTIKLI.

##essiz ürün sayısı nedir?

df["Description"].nunique()
df["Description"].value_counts().head()
df.groupby("Description").agg({"Quantity": "sum"}).head().sort_values("Quantity", ascending= False).head()
df["Invoice"].nunique()

###########Data Preparation

df.dropna(inplace=True)  # bos degerler silindi..

# iptal edilen siparişler vardı C ile baslıyordu fis numaraları. total priceta vs - deger veriyordu onlardan da kurtulalım.

df = df[~df["Invoice"].str.contains("C",na=False)]  # tilda işareti (~) değildir anlamına geliyor yani içinde C içermeyenleri listele.

####RFM METRIKLERININ HESAPLANMASI..........

# RECENCY,FREQUENCY,MONETARY
# recencyde tarihe göre sıralayacagız ama veriseti 2010-2012 biz 2023teyiz.

df["InvoiceDate"].max()

# son siparis tarihi 9 aralık 2010. mesela 2 gün sonrasına bugünmüş gibi diyelim ve ona göre işlem yapalım

today_date = dt.datetime(2011, 12, 11)
type(today_date)  # datetime typeında
type(df["InvoiceDate"])
df['InvoiceDate']= pd.to_datetime(df['InvoiceDate'])

# simdi rfm degeleri için bir groupby yapalım. customerid kırılımında va r f m degelerini bulalım. r degeri invoicedate f degeri invoice m degeri totalprice
df.head()
rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,
                                    "Invoice": lambda x: x.nunique(),
                                    "total_price": lambda x: x.sum()})

rfm.head()

# columnları degıselım.

rfm.columns = ["recency", "frequency", "monetary"]

# columnlar degisti.

rfm.describe().T

# monetary yani odenen parada mın 0 degerlerı var. istenmeyen bir deger. onları silelim.


rfm = rfm[rfm["monetary"] > 0]

rfm.shape

#####RFM SKORLARININ HESAPLANMASI...

# RECENCY TERS OLACAK UNUTMA!

df.head()
rfm.head()
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=([5, 4, 3, 2, 1]))

rfm["frequency_score"] = pd.qcut(rfm["frequency"], 5, labels=([1, 2, 3, 4, 5]))

# bunda hata verdi çünkü bu aralıkların bazıları boş kaldı. so sebeple frequency eklemek gerek

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=([1, 2, 3, 4, 5]))

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=([1, 2, 3, 4, 5]))

# bize RF lazım M ye gerek yok.

##Simdi RFM Skoru oluşturalım.


rfm["rfm_score"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

# stringe cevirdik ve topladık yani 55,23,12,52 gibi degerler olacak.


# champsları gorelim...

rfm[rfm["rfm_score"] == "55"]

####RFM SEGMENTLERINI OLUSTURUP ANALIZ EDELIM....

# REGEX YAPACAGIZ. 55= CHAMP YAZ GIBI KULLANACAGIZ.


# RFM ISIMLENDIRMESI..

rfm.head()
seg_map = {

    r"[1-2][1-2]": "hibernating",

    r"[1-2][3-4]": "at_risk",

    r"[1-2]5": "cant_loose",

    r"3[1-2]": "about_to_sleep",

    r"33": "need_attention",

    r"[3-4][4-5]": "loyal_customers",

    r"41": "promising",

    r"51": "new_customers",

    r"[4-5][2-3]": "potential_loyalist",

    r"5[4-5]": "champions",

}

# yeni bir segment olusturlalım ve bunları ekleyelim.

rfm["segment"] = rfm["rfm_score"].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# segmente göre groupby yaptık ve her segmenttekilerin mean ve count degerlerını bulduk.

# sırket bızden ornegın kaybetmemız gerekenlerı lıstele dedı.

rfm[rfm["segment"] == "need_attention"].head()

# bunların id leri..

rfm[rfm["segment"] == "need_attention"].index

# ekip bizden bir excel ya da csv olarak istese..

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "need_attention"].index

new_df.to_csv("new_customer.csv")

# ya da hepsini istiyorlarsa

rfm.to_csv("rfm.csv")

#########TÜM SÜRECİN FONKSİYONLAŞMASI(BİR SCRIPTE CEVIRILMESI))

# bu zamana kadar yazılanları bir functa toplayalım. hepsini tek bir func da yazabiliriz ama her step için ayrı ayrı da yazabiliriz.

df.head()
def create_rfm(dataframe, csv=False):
    # Veriyi Hazırlama

    dataframe["total_price"] = dataframe["Quantity"] * dataframe["Price"]

    dataframe.dropna(inplace=True)

    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI

    today_date = dt.datetime(2011, 12, 11)
    dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate'])
    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,

                                               "Invoice": lambda x: x.nunique(),

                                               "total_price": lambda x: x.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]

    rfm = rfm[(rfm["monetary"] > 0)]

    # RFM SKORLARININ HESAPLANMASI...

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=([5, 4, 3, 2, 1]))

    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=([1, 2, 3, 4, 5]))

    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=([1, 2, 3, 4, 5]))

    rfm["rfm_score"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

    ##Segmentlerin İsimlendirilmesi

    seg_map = {

        r"[1-2][1-2]": "hibernating",

        r"[1-2][3-4]": "at_risk",

        r"[1-2]5": "cant_loose",

        r"3[1-2]": "about_to_sleep",

        r"33": "need_attention",

        r"[3-4][4-5]": "loyal_customers",

        r"41": "promising",

        r"51": "new_customers",

        r"[4-5][2-3]": "potential_loyalist",

        r"5[4-5]": "champions",

    }

    rfm["segment"] = rfm["rfm_score"].replace(seg_map, regex=True)

    rfm = rfm[["recency", "frequency", "monetary", "segment"]]

    rfm.index = rfm.index.astype(int)  # indexi int yaptık

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm




rfm_new = create_rfm(df, csv=True)

df = df_.copy()

# ekip bizden bir excel ya da csv olarak istese.. dfnin ilk halini tekrar alalim

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "need_attention"].index

new_df.to_csv("new_customer.csv")

# ya da hepsini istiyorlarsa

rfm.to_csv("rfm.csv")


#########################Müşteri Yaşam Boyu Değeri (Customer LifeTime Value CLTV)
###musteri yasam boyu degeri nasıl hesaplanabilir?

# bir cok yol olmasına ragmen bir ornek verelim.. satın alma basına ortalama kazanc * satın alma sayısı...

# CLTV(CUSTOMER LIFETIME VALUE) = (Customer Value/Churn Rate)x Profit Margin

# Customer Value = Average Order Value * Purchase Frequency

# Average Order Value = Total Price/ Total Transaction

# Purchase Frequency= Total Transiction/Total Number of Customer

# Churn Rate=1-Repeat Rate

# Repeat Rate = Birden fazla alışveriş yapan müşteri Sayısı/ Tüm Müşteriler

# Profit Margin = Total Price * 0.10

# bir ornekle pekistirelim.


# total number of customers:100

# churn rate: 0.8

# profit: 0.10

# islem(transaction): ucret (price)

# 1                    300

# 2                     400

# 3                    500

# total 3               1200


# avarage order value : 1200/3 = 400

# purchase frequency : 3/100

# profit margin:

# customer value : 1200/3 * 3/100 = 12

# cltv: 12/0.8 * 120 = 1800

# sonuc olarak her bir müsteri icin cltv degerlerine göre bir sıralama yapılıp belirli yerlerden bolunerek olusturulan gruplarla musterileri segmentleyebiliriz.

##########Data Prep

df_ = pd.read_csv("DERSLER/CRM/online_retail_II.csv")
df = df_.copy()
df.head()
df["total_price"] = df["Price"] * df["Quantity"]
df = df[~df["Invoice"].str.contains("C",na=False)]  # tilda işareti (~) değildir anlamına geliyor yani içinde C içermeyenleri listele.
df= df[(df["Quantity"]>0)]
df.dropna(inplace=True)

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x : x.nunique(),
                                         "Quantity" : lambda x:x.sum(),
                                        "total_price": lambda x:x.sum()})
cltv_c.columns=["total_transaction","total_unit","total_price"]
cltv_c.head()
##aslında RFM E benziyor fakat burada R YOK. F=TOTAL_TRANS M=TOTAL_PRİCE

####Ortalama Sipariş Değeri (avg. order value)
#avg. order value: total_price/total_transaction

cltv_c["avg_order_vale"]=cltv_c["total_price"] /cltv_c["total_transaction"]

#purchase frequency (satın alma sıklığı)
#purchase_freq = total_transaction/total_numb_of_customers

cltv_c["total_transaction"]
cltv_c.shape #5881 kişi oldugunu söylüyor  zaten.

cltv_c.head()

cltv_c["purchase_freq"] = cltv_c["total_transaction"] / cltv_c.shape[0]

#repeat and churn rate (tekrarlama ve kaybetme oranı)
#repeat rate: birden fazla alısveris yapan musteri / tüm musteri
#churn rate: 1-repeat rate


cltv_c[cltv_c["total_transaction"]>1] #4255 adet
cltv_c #5881
repeat_rate = cltv_c[cltv_c["total_transaction"]>1].shape[0]/cltv_c.shape[0]
cltv_c.drop(columns="repeat_rate",inplace=True)
churn_rate = 1 - repeat_rate

#Profit margin:
#profit margin : total_price* 0.10 (bir katsayıdır)

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

#Müşteri Değeri (Customer Value)
# Customer Value : avg_order_value * purchase_freq

cltv_c["customer_value"] = cltv_c["avg_order_vale"] * cltv_c["purchase_freq"]

#customer lifetime value (musteri yasam boyu degeri)
#customer liftime value: (customer_value / churn rate) * profit margin
cltv_c["cltv"] = (cltv_c["customer_value"] /churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv",ascending=False).head()
cltv_c.describe().T

#####Segmentler oluşturup Analiz YAPALIM.

#"cltv" column zaten en kapsamlı katsayı.
cltv_c.sort_values(by="cltv",ascending=False).head() #en onemli 5 tane musteriyi listeleyecek.
# 4 gruba ayıralım mesela cltvyi.

cltv_c["segment"]=pd.qcut(cltv_c["cltv"],4,labels=["D","C","B","A"]) #qcut kucukten buyuge sıralama yapıypor yani en ezikler a en iyiler d olacak.

cltv_c.sort_values(by="cltv",ascending=False).head() #en onemli 5 tane musteriyi listeleyecek.
#simdi şu a,b,c,d segmentlerini bi groupby yapalım


cltv_c.groupby("segment").agg({"count","mean","sum"})

##bu calısmalar sonunda bunu bir csv dosyasına cevirmek istersek..
cltv_c.to_csv("cltv_c.csv")

########Tüm sürecin Fonksiyonlaştırılması...

def create_cltv_c(dataframe, csv=False, profit=0.10):
    # Veriyi Hazırlama

    dataframe["total_price"] = dataframe["Price"] * dataframe["Quantity"]
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C",na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)

    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                                   "Quantity": lambda x: x.sum(),
                                                   "total_price": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # RFM METRIKLERININ HESAPLANMASI

    today_date = dt.datetime(2011, 12, 11)
    dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate'])
    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,

                                               "Invoice": lambda x: x.nunique(),

                                               "total_price": lambda x: x.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]
    # avg. order value: total_price/total_transaction

    cltv_c["avg_order_vale"] = cltv_c["total_price"] / cltv_c["total_transaction"]
    # purchase_freq = total_transaction/total_numb_of_customers

    cltv_c["total_transaction"]
    cltv_c.shape  # 5881 kişi oldugunu söylüyor  zaten.

    cltv_c.head()

    cltv_c["purchase_freq"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    # repeat and churn rate (tekrarlama ve kaybetme oranı)
    # repeat rate: birden fazla alısveris yapan musteri / tüm musteri
    # churn rate: 1-repeat rate

    cltv_c[cltv_c["total_transaction"] > 1]  # 4255 adet
    cltv_c  # 5881
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # Profit margin:
    # profit margin : total_price* 0.10 (bir katsayıdır)

    cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

    # Müşteri Değeri (Customer Value)
    # Customer Value : avg_order_value * purchase_freq

    cltv_c["customer_value"] = cltv_c["avg_order_vale"] * cltv_c["purchase_freq"]

    # customer lifetime value (musteri yasam boyu degeri)
    # customer liftime value: (customer_value / churn rate) * profit margin
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
    #segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_c


df= df_.copy()
clv = create_cltv_c(df)

##############MUSTERI YASAM BOYU DEGERI TAHMINI....
#AMAC YUKARIDA YAPTIGIMIZ YASAM BOYU DEGERINI BIR OLASILIK GIBI YAZIP TOPLU BİR DFTEKİ KİŞİLERİ BİR KİŞİ GİBİ YAPIP
#ONA GÖRE TAHMIN YAPABILMEK. DISARIDAN INPUTLA O KISININ TAHMININI YAPABILMEK VSVS...

#BG/NBD(BETA GEOMETRIC/ NEGATIVE BINOMINAL DISTRIBUTION) KINDA EXPECTED NUMBER OF TRANSACTION

#Customer Value = Purchase Freq * Average Order Val
#CLTV= Expected Nu of Transac * Exp. Avg. Prof
#CLTV = BG/NBD Model * Gamma Gamma Submodel

#bg/nbd diger adıyla buy till you die.
#buy(transaction process): bir müsteri alive oldugu sürece kendi transaction ratei etrafında rastgele satın almaya devam edecektir.
#transaction rate'ler her bir müşteriye göre değişir ve gamma dagılır.(r,a)
#dropout process(till you die)
#bir müsteri aliveris yaptıktan sonra belirli olaslıkla drop olur.
#dropout rateler her bir müsteriye göre degisir ve tüm kitle için beta dagılır.(a,b)
#bg-nbd modelinin formulasyonu...
#E(Y(t)|X = x,Tx, T, r, alpha, a,b) =

#x= tekrar eden satıs sayısı
#tx=recency degeri
#T = musterinin ilk alısveris yaptıgı tarihten bugunun cıkarımı (musteri yası)
#r a = gamma dagılımı ifadeleri
#a,b = inaktif olasılıgı(droprate), beta dagılımı ifadeleri

#biz zaten funclarla tahmin edecegiz..


#GAMMA GAMMA SUBMODELİ KINDA EXPECTED AVERAGE PROFIT

#Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.
#bir müsterinin islemlerinin parasal degeri(monetary) transaciton valuelarının ortalaması etrafında rastgele dagılır.
#ortalama transaction value, zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı için değişmez.
#ort trans. val. tüm müşteriler arasında gamma gamma dağılır.

#E(M|p,q,y,mx,x)=kişi ve dagılım ozelligi girilince ortalama ne kadar karlılık bırakacagını söyleyecegim diyor.

##cltv protection crmin anası-bsbasıdır.

###BG-NBD VE GAMMA GAMMA İLE CLTV PREDiCTiON

#VERİNİN HAZIRLANMASI
#BG-NBD MODELİ İLE EXPECTED NUMBER OF TRANSACTİON
#GAMMA-GAMMA MODELİ İLE EXPECTED AVERAGE PROFİT
#BG-NBD VE GAMMA GAMMA İLE CLTVNİN HESAPLANMASI
#CLTVYE GÖRE SEGMENT OLUSTURMA
#BU CALISMANIN FONKSIYONLASMASI


#############VERİNİN HAZIRLANMASI

#!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x: "%.4f" % x)
from sklearn.preprocessing import MinMaxScaler
df_ = pd.read_csv("DERSLER/CRM/online_retail_II.csv")
df = df_.copy()
#su an amac aykırı degeleri elemek....
def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01) #normalde 1 ve 4uncu ceyrek 0.25 0.75 olmalı. bu veri setine 0.01 ve 0.99 hocanın uygun gordugu degerler.
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range= quartile3 -quartile1
    up_limit= quartile3 + 1.5 *interquantile_range
    low_limit= quartile1- 1.5*interquantile_range
    return low_limit,up_limit

def replace_with_threshold(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit #eger low_limden dusuk deger varsa onu low_limite esitle.
    dataframe.loc[(dataframe[variable]>up_limit),variable] = up_limit #eger up_limden yuksek deger varsa onu up_limite esitle.


####DATA READING
df_.head()
df= df_.copy()
df.describe().T
#########veri ön işleme...
df.describe().T #- degerler var.
df.isnull().sum() #cok null deger var silinecek.
df.dropna(inplace=True)
df.describe().T #hala - deger var
df.isnull().sum() #0 oldu
df = df[~df["Invoice"].str.contains("C",na=False)]  #c leri sildik.
df=df[df["Quantity"]>0]
df=df[df["Price"]>0] #HER İHTİMALE KARSI NEGOLARI SILDIK

df.describe().T
#- ler gitti. ama 75lik çeyrekte quantity 12 olanın maxı 81k. demekki burada bir aykırı değer var.

#aykırı degeleri belirledigimiz threshold degerlerle degistirelim. daha once yazdıgımız func.

replace_with_threshold(df,"Quantity")
replace_with_threshold(df,"Price")

df.describe().T #max degerler baya degisti.düstü. baya bir traslama yaptık. ucundan.

#herbir ürüne kaç para ödendi?
df["total_price"]= df["Quantity"] * df["Price"]

#gunumuz tarihi problem olacak. veri setindeki son günden bir iki gün sonrayı bugünün tarihi gibi alalım.

today_date =dt.datetime(2011,12,11)


###########LifeTime Veri Yapısının Hazırlanması...
#gamma-gamma ve bg-nbd bizden bazı hazırlıklar istiyor.

#recency: bildigimiz yukarılardaki gibi degil de, son satın alma tarihi üzerinden geçen zaman.(son satın alma - ilk satın alma.) (haftalık istiyor)
#T: musterinin yasi. (Haftalık). analiz tarihinden todaydateten ne kadar süre önce ilk alışverişi yaptıgı.
#Frequency: tekrar eden satın alma sayısı (f>1)
#monetary: satın alma bşaına ortalama kazanç(toplam kazançtı normalde. burada ortalamya döndü)
df.head()
df["InvoiceDate"]= pd.to_datetime(df["InvoiceDate"])
cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x : (x.max()- x.min()).days,
                                                         lambda x : (today_date - x.min()).days],
                                        "Invoice" : lambda x:x.nunique(),
                                        "total_price": lambda x:x.sum()})
#cıktı da columnlar cirkin oldu 2 tane column ilki cok gereksiz onu yok edelim.

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["Recency","T","Frequency","Monetary"]
cltv_df
#simdi bizden istenilen rec,t haftalıga cevirme, freq > 1 olma, monetary ise ortalama kazanc onları halledelim.

cltv_df["Recency"] =cltv_df["Recency"] /7
cltv_df["T"] =cltv_df["T"] /7
cltv_df =cltv_df[cltv_df["Frequency"] > 1 ]
cltv_df["Monetary"] =cltv_df["Monetary"] / cltv_df["Frequency"]

cltv_df.head()
cltv_df.describe().T

######BG-NBD Modelinin Kurulması(satın alma sayısını modeller):

bgf=BetaGeoFitter(penalizer_coef=0.01) #penalizer_coef uygulanacak hata katsayısı.
#istenilenleri fitleyelim.
bgf.fit(cltv_df["Frequency"],cltv_df["Recency"],cltv_df["T"])

##########1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir???
bgf.conditional_expected_number_of_purchases_up_to_time(1,cltv_df["Frequency"],cltv_df["Recency"],cltv_df["T"]).sort_values(ascending=False)
#1 haftalık dediğimiz için ve bizim birimler haftalık bazda oldugu icin 1 yazdık. 1 ay deseydik, 4 yazacaktık. 3 ay 12 vsvs.

##bu fonks kısa adı da var predict. 1 haftalık ve 3 aylık beklenen kazancları df e ekleyelim.
cltv_df["1_week_expected_purc"]=bgf.predict(1,cltv_df["Frequency"],cltv_df["Recency"],cltv_df["T"])
cltv_df["3_months_expected_purc"]=bgf.predict(12,cltv_df["Frequency"],cltv_df["Recency"],cltv_df["T"])

####3 ayda sirketin toplam beklenilen satışı??
bgf.predict(12,cltv_df["Frequency"],cltv_df["Recency"],cltv_df["T"]).sum()

########### TAHMİN SONCULARININ DEGERLENDIRILMESI....
plot_period_transactions(bgf)
plt.show(block=True) #grafige göre yorum yapabiliriz. real vs model karsılastırması..

####################Gamma-Gamma Modelinin Kurulması(average profiti modeller)..

ggf= GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["Frequency"],cltv_df["Monetary"])

ggf.conditional_expected_average_profit(cltv_df["Frequency"],cltv_df["Monetary"]).head()
ggf.conditional_expected_average_profit(cltv_df["Frequency"],cltv_df["Monetary"]).sort_values(ascending=False)
cltv_df["expected_average_profit"]=ggf.conditional_expected_average_profit(cltv_df["Frequency"],cltv_df["Monetary"])
cltv_df.sort_values("expected_average_profit",ascending=False)


cltv_df.drop("expected_average_prfofit",axis=1)



#######BG-NBD VE GG MODELİ İLE CLTVNİN HESAPLANMASI.............

cltv= ggf.customer_lifetime_value(bgf,cltv_df["Frequency"],cltv_df["Recency"],cltv_df["T"],cltv_df["Monetary"], time = 3, freq="W",discount_rate=0.01)
#istenilen degerler freq rec t ve monetary idi. Time burada kaç aylık cltv tahmini istiyorsun. 3 ay dedik. freq= bizim calıstıgımız deger. biz haftalık calıstık "W". discount_rate= bu sürecte indirim YAPAR MISIN? onun oranı?)
cltv
#indexi düzeltelim.
cltv=cltv.reset_index()
cltv.head()
#cltvyi ana veri setine ekleyelim. ortak column cust id

cltv_final = cltv_df.merge(cltv,on="Customer ID",how="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)

#cltv degerlerine ulastık. buradan cok güzel sonuclar cıkarabiliriz.
#biz receny yüksekse kötü demiştik. ama bir müsteri uzun zamandır bizdeyse yani recency yuksek ama aynı zamanda freq vs yuksekse bu adamlar uzun zamandır alısveris yapmıyor ama her an yüklü bi şey gelebilir gibi düşünülür. yani cltvsi en yüksek bunlar olur.
#bazılarının freq cok yuksek ama monetary dusuk. bunlar sürekli alışverişci oldugu için bunu da ön planlara cıkardı...

####Musteri Segmentleri Olusturma


cltv_final["segment"] = pd.qcut(cltv_final["clv"],4,labels=["D","C","B","A"])
cltv_final.head()

cltv_final.groupby("segment").agg({"sum","mean","count"})


#ve bunları fonksiyonlaştıralım...

def create_cltv_p(dataframe, month=3):
    dataframe.dropna(inplace=True)
    dataframe["total_price"] = dataframe["Price"] * dataframe["Quantity"]
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C",na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe = dataframe[(dataframe["Price"] > 0)]
    replace_with_threshold(dataframe,"Quantity")
    replace_with_threshold(dataframe,"Price")
    today_date = dt.datetime(2011, 12, 11)
    dataframe["InvoiceDate"]= pd.to_datetime(dataframe["InvoiceDate"])
    cltv_df = dataframe.groupby("Customer ID").agg({"InvoiceDate": [lambda x : (x.max()- x.min()).days,
                                                         lambda x : (today_date - x.min()).days],
                                             "Invoice" : lambda x:x.nunique(),
                                             "total_price": lambda x:x.sum()})
    cltv_df.columns = cltv_df.columns.droplevel(0)

    cltv_df.columns = ["Recency", "T", "Frequency", "Monetary"]
    cltv_df["Recency"] = cltv_df["Recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7
    cltv_df = cltv_df[cltv_df["Frequency"] > 1]
    cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]


    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"])
    cltv_df["1_week_expected_purc"] = bgf.predict(1, cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"])
    cltv_df["3_months_expected_purc"] = bgf.predict(12, cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"])
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                                                                 cltv_df["Monetary"])


    cltv = ggf.customer_lifetime_value(bgf, cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"], cltv_df["Monetary"],
                                       time=month, freq="W", discount_rate=0.01)
    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])


    return cltv_final

df= df_.copy()
create_cltv_p(df,3)


###quizzz

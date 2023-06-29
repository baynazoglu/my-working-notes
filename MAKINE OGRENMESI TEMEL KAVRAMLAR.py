##Makine Öğrenmesi Nedir?
#Pcnin insanlar gibi benzer şekilde öğrenmesini sağlamak için çeşitli algoritma ve tekniklerin geliştirilmesi için çalışan
# bilimsel çalışma alanı.

#Örneğin titanic verisini düşünelim. Bize bir kişi verildiginde hayatta kalıp kalmamasına nasıl bakarız? yaşına cinsiyetine
# ve kaçıncı sınıf olduguna bakarız. kafamız böyle calısır. mlde de bu kafaya ulasmaya calısacagız,

##Değişken Türleri:
#sayısal degiskenler:
#kategorik degiskenler(nominal,ordinal):
#bagımlı degisken(target,output,dependent,response) ölüm, ev fiyat tahmininde fiyat, diabetesde diabet olma durumu vs.
#bagımsız degisken: bagımlı degiskeni olusturdugunu düsündügümüz, hedefimizi etkileyen degiskenler.


##Öğrenme Türleri:
#1.Denetimli Öğrenme(Supervised Learning): pratikte en çok karşımıza çıkan
#2.Denetimsiz Öğrenme(Unsupervised Learning):
#3.Pekiştirmeli Öğrenme(Reinforcement Learning):

#1.Pekiştirmeli Öğrenme: Deneme yanılma yöntemiyle öğrenme. Bir çocugun sobaya dokunmamayı elini yakınca ögrenmesi gibi.
# autodriver Virajı dönerken fren yapmayı,direksyonu kullanmayı deneme yanılmayla ogrenir.

#2.Denetimli Öğrenme:Üzerinde calısan veride bir target,dependent varsa bu denetimli ögrenmedir. Dependentin denetiminde ögrenmedir.

#3.Denetimsiz Ogrenme: Dependent yoksa verisetinde.


##Problem Türleri:
#REGRESYON PROBLEMLERİ:EĞER BAGIMLI DEGISKEN SAYISALSA BU REGRESYON PROBLEMİDİR.
#SINIFLANDIRMA PROBLEMİ: BAGIMLI DEGISKEN BIR KATEGORİK DEGİSKENSE.(SAYISAL OLARAK YAZILABİLİR AMA KAT.SE)
#(ORN:CHURN OLMA 1VE0 AMA CATEGORİKTIR CUNKU YES VE NODUR AMA ML İCİN CAT YAPMISLARDIR.)



#MODEL BASARI DEGERLENDIRME YONTEMLERI:
#TAHMINLERIM NE KADAR BASARILI?
#REGRESYON MODELLERINDE BASARI DEGERLENDIRME:
#MSE(MEAN SQUARE ERROR):HATA KARELER ORTALAMASI. ADIN SOYADIN GIBI BIL.
 #KURMUS OLDUGUMUZ MODEL ARACILIGIYLA BULDUGUMUZ DEGERLERLE BAGIMLI DEGISKENIN KENDI DEGERLERI ARASINDAKI FARKIN KARESI TOPLAMININ/BUTUN GOZLEM BIRIMINE BOLUMU.

#RMSE:ROOT MEAN SQUARE ERROR: MSE NIN ROOTA ALMA İŞİ.

#MAE: MUTLAĞA ALMA İŞİ. KARESİ YERİNE MUTLAK ORJ DEGER- BULDUGUMUZ DEGER GİBİ. YİNE AYNI
#######

#SINIFLANDIRMA MODELLERINDE BASARI DEGERLENDIRME:
#ACCURACY: DOGRU SINIFLANDIRMA SAYISI / TOPLAM SINIFLANDIRMA SAYISI. 10 GOZLEMDEN 9U DOGRUYSA %90 BASARI ORANI.


#MODEL DOGRULAMA YONTEMLERI: KULLANDIGIMIZ MODELIN NE KADAR DOGRU OLDUGUNU OGRENMEK ICIN.
#HOLDUT YONTEMI:(SINAMA SETI YONTEMI)
#ORİJİNAL VERİYİ BÖLERİZ 2 PARÇAYA. EĞİTİM SETİ VE TEST SETİ OLMAK ÜZERE. EGİTİM SETİNDE ZATEN MODELİ ÖGRENMİS OLUR AMA
#TEST SETİNDE BİR BİLGİSİ OLMAYACAĞI İÇİN ONLA TEST EDERİZ.

#K KATLI CAPRAZ DOGRULAMA(K FOLD CROSS VALIDATION)
# holdout sınama setinde orj veri setini test ve egitim olarak 2ye bolmustuk. orn %80i egitime %20si teste. sonra bu testteki
# degerlerden bi dogrulama yapmıstık. orn. 100 verim var ve 20 tanesini teste ayırdık. 20 az bi sayı. bu 20değer eğitimdekilere kıyasla
# daha farklı degerler de olabilir. bize hatalı bi sonuc gosterebilir. cunku veri az. bu sebeple katlı çapraz cross validation daha mantıklı
# ***EĞER VERIMIZ COKSA %20LIK DILIM  YETERLI OLABILIR YANI HOLDOUT YETERLI OLABİLİR AMA AZ OLMASI DURUMUNDA KATLI CAPRAZ KFOLD DAHA MANTIKLI GİBİ
# GİBİ DEDİK CUNKU MAKINE OGRENMESINDE BIR KESINLIK YOKTUR. PEKI NASIL YAPILIR BU CROSS VALIDATION?
# ORIJINAL VERIYI 5 ESIT PARCAYA BOL MESELA. ONCE ILK4U EGIT 1INI TESTE SOK. SONRA 2-5 EGIT 1I TESTE SOK, 1-345I EGIT 2YI SOK VS. HEPSININ
# KOMBINASYONLARINI YAP SONRA TESTTEN CIKAN SONUCLARIN ORTALAMASINI AL. YA DA
# YINE HOLDOUT GIBI EGITIM VE TESTE BOL. EGITIME YUKARIDAKI GIBI CAPRAZ UYGULA 5E BOL 4UNU EGIT 1INI TESTE SOK VS EN SON TESTI DE TESTE SOK.


##YANLILIK-VARYANS DEGIS TOKUSU(BIAS-VARIANCE TRADEOFF):ASIRI OGRENME OVERFITTING.
#MODELIN VERIYI OGRENME ISI. BIZ MLIN MODELIN ORUNTUSUNU,YAPISINI OGRENMESINI ISTIYORUZ.VERIYI OGRENMESINI DEGIL
#UNDERFITTING:AZ OGRENME ISI.
#DOGRU MODEL: DUSUK YANLILIK, DUSUK VARYANS..
 #ASIRI OGRENMEYE DUSTUGUNUZU NASIL ANLARSINIZ?:
#TAHMIN HATASI V MODEL KARMASIKLIGI GRAFIGI DUSUNELIM. EGITIM VE TEST SETI ICIN DEGERLER OLSUN. DOGRU ORANTI VARKEN SIKINTI YOK.
# NE ZAMAN TEST SETINDE TAHMIN HATALARI ARTARKEN, EGITIM SETINDE DUSMEYE BASLIYORSA(HATA), ASIRI OGRENME BASLAMIS DEMEKTIR.
#MODEL KARMASIKLIGI: AGACTA,LİNEERDE VS DEGİSİR. MESELA MODELI DAHA DA KARMASIKLASTIRMAK. LINEERE USTEL FUNC EKLEMEK GIBI. AGACA 18 AGAC EKLEMEK GIBI
# BASITKEN DOGRU ORANTI VARDI KARMASIKLASMAYA BASLADI VE OVERLEARNING OLDU. GEREK YOK. BI NOKTADAN SONRA DURMAK LAZIM
#







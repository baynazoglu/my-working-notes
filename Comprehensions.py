##########################################List Comprehensions(ÇOK ONEMLI)
##birden fazla satır ve kodla yazılan işlemleri tek bir satır ve kodda yazabilme işlemidir.
#old-school yazalım
salaries=[10000,20000,30000,40000]
def new_wages(x):
    return x*1.2 #20 zamlı
for i in salaries:
    print(new_wages(i))

#bunu boş bi listeye eklemek istesek:
a=[]
for i in salaries:
    a.append(new_wages(i)) #yazmamız gerekir.
#maaşı 30k altına 20, 30k üstüne 10 zam desek if-else case yazmamız gerekir.

##List-Comprehensions:

[new_wages(i*1.2) if i < 30000 else new_wages(i*1.1) for i in salaries]

[salary *2 for salary in salaries] #maaşları 2x yaptık.

[salary *2 for salary in salaries if salary < 30000] #30kdan düşük maaşları 2x yaptık.

[salary *2  if salary < 30000 else salary*0.95 for salary in salaries ]
#30kdan düşüklere 2x, diğerlerine yüzde 5 indirim yaptık, if-else birlikte olunca fordan önce yazıldı.
[new_wages(salary*2) if salary < 30000 else new_wages(salary*1.85) for salary in salaries]
#var olan new_wages(maaşa 20 zam) fonksiyonuna, maaşları 2x yapıp soktuk eğer 30kdan düşükse,değilse yüzde 85li halini soktuk.

#istenilmeyen students_n kücük harfle, diğerleri büyük harfle yazılsın.
students=["Ayshe","Fatma","Ali","Veli"]
students_n=["Ayshe","Ali"]
[i.lower() if i in students_n else i.upper() for i in students]

[i.upper() if i not in students_n else i.lower() for i in students]


###################################Dict Comprehension

#listelere benzerdir.

dict={"a":1,
      "b":2,
      "c":3}  #keys(),values(),items() dict methodları.
dict.values()
#her bir degerin karesini al.
{k:v**2 for (k,v) in dict.items()}
#k keys v values, k sabit. vnin karesi.

##keylere bir işlem yapmak istersek.
{k.upper():v for (k,v) in dict.items()}

#############UYGULAMA-MULAKAT DICT COMPREHENSIONS
#AMAÇ: sayıların oldugu listeden, çiftlerin karesi alınıp bir sözlüğe eklensin.
#keyler orijinal, valuelar degistirelen değerler olsun.
numbersss=range(7)
numberss=[2,3,4,5,6,7,8,9]
new_dict={}

{i:i**2 for i in numberss if i % 2==0} #list compr

for i in numberss:
    if i %2 == 0:
        new_dict[i]= i**2   #old-school way.

######List&Dict Comprehensions Uygulamaları

#Bir veri setindeki değişken isimlerini değiştirmek.(methodsuz)

#before=["total","value","is","too","much"]
#after=["TOTAL","VALUE","IS","TOO","MUCH"]

import seaborn as sns
df=sns.load_dataset("car_crashes")
df.columns
#bu columnları buyutecegız...
A=[]
for i in df.columns:
     #print(i.upper())  #old-schoolic way, ve kalıcı değil anlık yazdırdık.
    A.append(i.upper()) #still old-schoolic but A listesine ekledik.
#df in columnslarını büyütmek istiyoruz..
df.columns= A

#list compr:
df=sns.load_dataset("car_crashes")
df.columns=[i.upper() for i in df.columns]      #list compr
df.columns

## ISMINDE "INS" olanlara flag olmayanlara no flag ekle.
df.columns=["FLAG_" + i if "INS" in i else "NO_FLAG_" + i for i in df.columns]      #list compr
df.columns

#stringleri ekleme işi : "a" +"b" = ab


####### (ÇOK DEĞERLİ) Amaç keyi string, valuesu asagıdaki gibi liste olan sozluk olustur. sayısal deger olanlardan sadece. kategoriğin mini maxı olmaz.

##Output:
{"total":["mean","max"],
 "speeding": ["max","var"]}

import seaborn as sns
df=sns.load_dataset("car_crashes")
df.columns
#keyler df columns olacak.
#valueları int  olanları seçecek bir kod;
num_cols=[i for i in df.columns if df[i].dtype != "O"]
#df deki i değerinin dtypeı object olmazsa yazdır dedik.
dicti={}
agg_list=["mean","max","min","sum"]
for i in num_cols:
    dicti[i]=agg_list
dicti

###that was old-school way.

new_dicti={i:agg_list for i in num_cols}  #dict compreh. way.

##bu neden önemli?? asagıda ne kadar kolaylatırdıgını gorelım.

df[num_cols].agg(new_dicti) #valuelardaki tüm fonks. cevaplarını tek kodda bulduk.
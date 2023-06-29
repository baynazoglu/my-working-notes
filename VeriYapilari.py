###################################Sayılar
a = 5  #int
b = 10.5  # sayı ama float
a / 3
b * 4
a ** 2 #karesi
# float to intiger Tipleri Değiştirmek
int(b)
float(a)
int(a * b / 10)

################################Strings(Karakterler)
print("John")
a="John"
type(a)
##Çok satırlı karakter Dizileri
"""selamlar herkese """
##Karakterlerde Elemanlara Erişmek
a[0]
##Karakterlerde Slice(Dilim] İşlemleri
a[0:2]
##Karakterlerde Eleman Sorgulama
"jo" in a #ptyhon büyük harf hassas oldugu icin false dedi
"Jo" in a
"Jn" in a
print("herkese \nselamlar") # \n bir satır asagı ındırdı.

################################String Methods
dir(str) # str le alakalı methodları gosterir.
##len
type(a)
len(a)
##ctrl + methoda,fonksiyona tıklarsan acıklama sayfası acılır.

##Upper() Lower()

"herkese selamlar".upper()
##pythonda tek tek tanıtmaya gerek yok. "herkese selamların str oldugunu hemen anladı.
"HERKESE sElamLar".lower()

##replace: karakter degistirmek icin
"herkese merhabalar".replace("e","a") #e leri a yaptı.

##split: böler

"iyi ki buradayız".split() #bu custom degeri.

##strip: ön tanımlı kıpma
"hahahaha".strip()
"hahahaha".strip("a")

##capitalize() startswith("")

################################Listeler

#Değiştirilebilirler. Sıralıdır.İndex işlemi yapılabilir. Kapsayıcıdır.

notes=[1,2,3,4,5]
type(notes)
names=["ali","veli","["ayshe","ayse"]","1",True]
type(names) #kapsayıcıdır. liste içinde int bool str var

names[2]
names[2][1] #liste içi listeye eriştik. listeler sıralıdır,index işlemi yapılabilir

notes[0]
notes[0]=2
notes #degistirilebilirler.

###############################Liste Methodları

dir(notes)

##append:eleman ekler, len():alreadyknewit.,pop:indexe göre siler, insert:indexe ekler
len(notes)
notes.append(12)
notes
notes.pop(0)
notes.insert(0,12)


##########################################Dictionary
##key&value, değiştirilebilir. sırasız, kapsayıcı.

dict= {"ankara":"06",
      "istanbul":"34",
      "izmir":"35"}
dict2 = {"ankara" :["yenimah","cank"],
         "istanbul" : ["beskt","kadik"]} ##kapsayıcıdır list str var icinde

dict["ankara"]
dict2["ankara"][1] #cankayaya ulastık

##key sorgulama
"ankara" in dict #true
"AnKara" in dict #false

#key'e göre value ulasmak.
dict.get("izmir")
dict["izmir"]

dict["izmir"] = 37 #degistirebilirdir. izmir artık 37.

##Tüm keylere erişmek.
dict.keys()
##tüm valuelara erişmek
dict2.values()

#tüm çiftlere tuple halinde ulasmak
dict.items()

##Yeni key value eklemek.
dict.update({"mersin" : "33"})

################################TUPLE(DEMETLERR)
#listenin aksisi. değiştirilemez. sıralı. kapsayıcı
demet= (1,2,"mark")
type(demet)
demet[0]
demet[0] = 99 #error verdi degistiremedik.

demet2=list(demet)
type(demet2) #su an degisebilir ooldu. ama artık tuple degil.

#################################SET (KUMELER]
#degistirebilir. sırasız+essiz, kapsayıcıdır.

##difference() iki küme farkı.
set1=set([1,2,3])
set2=set([1,3,5])

set1.difference(set2) #2yi verir. 1de olup 2 de olmayan.
set2.difference(set1)
#or
set2-set1
#symmetric_difference(): iki kümede de birbirlerine göre olmayanlar
set1.symmetric_difference(set2) #set2.sym(set1) ile aynıdır.

#intersection():kesisim.
set1.intersection(set2) #sıralamanın onemı yok set1set2, set2set1.
#or
set1 & set2
#union() iki küme birleşimi
set2.union(set1)

#isdisjoint(): iki küme kesişimi boş mu?

set1.isdisjoint(set2) #false cunku kesisimde 1,3 var.

#issubset: alt kümesi mi?, issuperset(): kapsıyor mu?

total=3.4+2.6
print(total)

#####################################################################QUIZ

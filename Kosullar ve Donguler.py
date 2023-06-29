# true-false
1 == 1
1 == 2

####################################IF&ELSE&ELIF

if 1 == 1:
    print("thats true")

if 1 == 2:
    print("thats true")
else:
    print("thats false")


# numara check etme işi functionı yazalım. DRY olduk

def numcheck(x):
    if x == 10:
        print("10 is the correct number")
    else:
        print("try again")


numcheck(10)


# 3 DURUMLU DÜŞÜN.. EŞİT OLMA BÜYÜK KÜCÜK OLMA DURUMU.
def numcheck(x):
    if x < 10:
        print(x, " is smaller than 10")
    elif x > 10:
        print(x, " is bigger than 10")
    else:
        print(x, " is equal to 10")


numcheck(11)

########################DONGULER
#################FOR
stud = ["ali", "ayshe", "fatma", "veli"]
stud[2]
# üzerinde iterasyon yapılanbiler üzerinde gezip,işlem yapma sağlar.

for ogr in stud:  # ogr: içinde her bir elemanı gezecek şeyin adı.
    print(ogr)

for ogr in stud:  # ogrencileri büyüttük.
    print(ogr.upper())

salaries = [10000, 20000, 30000, 35000]
for salary in salaries:
    print(int(salary * 1.2))  # maaslara yüzde yirmi zam.

# dry unutma. maaşa zam yapan fonksiyon yaz. yeni maas klasörüne ekle.
new_salaries = []


def new_salary(salary, rate):
    new_salaries.append(int(salary + (salary * rate) / 100))
    print(new_salaries)


new_salary(10000, 30)

# peki ya maaşı ben seçmesem de, var olan maaşlara istenilen oranda zam yapılsa?

for x in salaries:
    print(new_salary(x, 35))

# once fonksiyon yazdık ve dedik ki yeni maas hesaplama aracı. sonra for dongusuyle var olan maaslara bu fonksiyonu uyguladık.

# maaşı 30k üstü olanlara yüzde 20 zam,30k ve altındakilere yüzde 30 zam.

salaries
yeni_maas = []


def new_sal():
    for sal in salaries:
        if sal < 30001:
            yeni_maas.append(sal * 1.3)
        else:
            yeni_maas.append(sal * 1.2)
            print(yeni_maas)


new_sal()

########################UYGULAMA-MüLAKAT SORUSU
# Amaç: Aşağıdaki Şekilde String değiştiren fonksiyon yaz.
# before: "hi my name is john and i am learning python.
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

before = "hi my name is john and i am learning python."
after =[]

for i in before:
    print(before.index(i))

    if before.index(i) % 2 == 0:
        after.append(i.upper())
    else:
        after.append(i.lower())

print(after)

#######BURADA PATLADIK CUNKU A ILK INDEXINI ALDI VE SONRAKI A LAR DA AYNI NUMARADAN DEVAM ETTI.


before.index("i")
--------------------------------------------------
########Range Komutu

range(len(before))   #length of before :44 bu sebeple range [0:44] oldu.
for i in range(len(before)):
    print(i)
###burada istedigimiz indexlemeyi yaptık..

    def sentenceconverter(first_sentence):
        second_sentence = ""
        for i in range(len(first_sentence)):
            if i % 2 == 0:
                second_sentence += first_sentence[i].upper()
            else:
                second_sentence += first_sentence[i].lower()
        print(second_sentence)

                sentenceconverter("hi my name is john and i am learning python.")

## sadece o cümleye özgü değil genel bir function yazdık. for döngüsüyle girilen stringin indexlerinde gezdik. indexler tek sayıysa lower cift ise upper yaptık. ve onu boş strmize ekledik.

###################################BREAK WHILE CONTINUE
#akışı kesmeye veya atlayarak devam etmeye ya da koşula göre kesmeye yarar.

salaries
for salary in salaries:
    if salary == 30000:
        break  #durdur
    print(salary)  #30k gelene kadar yazdı ve durdu.


for salary in salaries:
    if salary == 30000:
        continue  #30kyı salla devam et.
    print(salary)  #30k dısındakileri yazdı.


number=5
while number < 10: #oldugu müddetçe..
    number +=1
    print(number)

#############################Enumerate: Oto counter/index ile for loop
#iteratiflere hem gezerken aynı zamanda index bilgisini tutar.

students=["Ali","Ayshe","Veli","Fatma","Zehra"]
#normal for dongusu:
for i in students:
    print(i)

#enum ile 2 tane olmalı. biri index biri dolasan.
for indeksi,i in enumerate(students):   #studentste index ve kendi degerleri itere et.
    print(indeksi,i)   #0 ali , 1 ayshe vsvs..

#enumerate(students) python direkt indexi 0 dan baslatır. 1den baslamak isteyseydik;

for indeksi,i in enumerate(students,1):
    print(indeksi,i) #1 ali 2 ayshe vsvs.

#cift indexi bi liste, tek indexi bi liste;
tek_indeksli=[]
cift_indeksli=[]
for indeksi,i in enumerate(students):
    if indeksi % 2 ==0:
        cift_indeksli.append(i)
    else:
        tek_indeksli.append(i)
print("tek indeksliler:",tek_indeksli,"\ncift indeksliler:",cift_indeksli)

#####Mulakat sorusu Enum
#divide_students fonksiyonu yazın. bir student listesi var.
#cift indexler bir, tek indexler bi listeye alın.
#fakat bu iki liste de tek bir liste olarak return olsun.
def divide_students(students):
total=[[],[]] #direkt indexe göre ekleyebiliriz.
for indeksi,i in enumerate(students):
    if indeksi % 2 ==0:
        total[0].append(i)
    else:
        total[1].append(i)
print(total)
return total (?????)

#########################Alternating Fonksiyonunu Enum ile yazma.
#tek indekslileri büyüt, çifleri küçült.
def alternating_withenum(old_cumle):
    new_cumle = ""
    for indeksi,i in enumerate(old_cumle):
        if indeksi % 2 == 0:
            new_cumle += i.lower()
        else:
            new_cumle += i.upper()
    print(new_cumle)

alternating_withenum("hi my name is john")
#enum yerine range(len(str)) yapabilirdik ama enum better.

##############################ZIP
#liste içinde tuple formunda ayrı listeleri zipleme işlemi.

students=["Ali","Ayshe","Veli","Fatma","Zehra"]
students2=["Ahme","Rehe","Teli","Etma","Mehra"]
salaries = [10000, 20000, 30000, 35000,45000]
total_stud=list(zip(students,students2,salaries))


###############################Lambda Map Filter Reduce

#pythonın bazı araçları, lambda cok onemli. kullan-at fonksiyondur.

def func(a,b)
    print(a+b)
#fonksiyon yazmak yerine lamda ile tekte halledebiliriz.
x=lambda a,b:a+b #yazabiliriz. normalde x= lambda diye bi kullanım olmaz applyla yazılır, ornek icin yazdık.

######map
#functiondaki for dongulerini ben yapabilirim = map.

def new_wages(x):
    return x*1.2 #20 zamlı
for i in salaries:
    print(new_wages(i))
##old school yazdık maple söyle olurdu.
list(map(new_wages,salaries)) # list olsun istedik, ilk degisken fonksiyon ikincisi ise nerede gezmesini istedigin liste.

#daha da simple yapalım func yerine lambda..

list(map(lambda x:x*1.2,salaries)) #func yerine lambda ile yazdık xe 20 zam yap. mapledi salariese.

##########filter: filtreler. az kullanılır.

list(filter(lambda x:x%3==0,salaries)) #map yerine filter ile yazdık. salariesteki maaşların 3e bölümü 0 olanları filtrelerdi.
# aslında body partındaki if else kısmını burada, filter ile yaptık.

########### reduce: indirgeyen..

from functools import reduce
list_store=[1,2,3,4,5]
reduce(lambda a,b:a+b, list_store)


################QUİZ

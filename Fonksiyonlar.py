###################Fonksiyonlara Giriş Ve Fonksiyon Okuryazarlığı#####################
##Fonksiyon OkurYazarlığı
?print #print func hakkında bilgi almak için,dokumantasyona ulasmak için. docstringine ama console da yazabilirsin.
#help(print) de docstring açar
help(print)

print("a","b")
print("a","b",sep="--") #ayırma aracını boşluktan iki cizgiyle degistirdik sep argumanıyla.

###Fonksiyon Tanımlama
def fonksiyon1 (arguman): #defle fonksiyon tanımlıyoruz fonksiyonumuzun adı fonksiyon 1. argumanı var mı? var evet "arguman" adı
    print(arguman*2)  #fonksiyonun body kısmında girilen degerin 2 katını cıkarma fonksiyonu yazdık.


fonksiyon1(5) #10 degerini verdi.
help(fonksiyon1)

##İki argümanlı bir fonksiyon tanımlama..

def func1 (x,y): #2 argüman yazdık x ve y.
    print(x+y)    #body kısmında iki argümanı topla dedik.

    func1(3,5) #8i verdi
    ##or
    func1(y=5,x=3) #yine 8i verdi.

#########################DOCSTRING (FONKSIYONLARA BILGI NOTU EKLENEN YER)
#yazdıgımız fonksiyona docstring ekleyelim..

def func1 (x,y):
     """
     Sum of two numbers
     :param x:  int,float
     
     :param y:  int,float   
     :return:   int,float
     """"
     # 3 tane tırnak koyup enterlayınca oto doldurdu. biz de doldurduk. int float yazdık daha her şeyi ekleyebilirsin ne tanıtmak istersen.
    print(x+y)

help(func1)
func1(50,52)

#########Fonksiyonların Statement(Body) Bölümleri

def func2 ():
    print("hi")
    print("hello")
    print("selam")
    func2() #3ünü de direkt yazdırdı.
def func3 (x):
    print(x)
    print("hello")
    print("selam")
    func3() #hata. x argumentini yazmadıgmız için.
    func3(3) #3,hello,selam.
#girilen 2 sayıyı çarp bunları bi nesnede tut ve yazdır.

def func4 (x,y):
    z=(x*y)
    print(z)

    func4(3,70)
#girilen değerleri bir liste içinde saklayacak fonksiyon.
list5=[]  #global scope
def func5 (x,y,z):
    list5.append([x,y,z])
    print(list5)                #body partındaki local scopelar
func5(1,2,7)
func5(2,2,3)

##################Ön tanımlı argümanlar/parametreler
def func3 (x="hi"):
    print(x)
    print("hello")
    print("selam")
func3("aynen")   #on tanımlı hi yerine aynen yazdık.#burada

##########Ne zaman Fonksiyon Yazılır??
#DRY prensibi. Dont Repeat Yourself.
#Örneğin bir şirket bir hesaplama yapıp veriyi çekip o veriyi sürekli hesaplıyor.

def calc(varm,mois,charge):
    print((varm+mois)/charge)
##artık her gelen degeri çat diye hesaplayacak. DRY
calc(12,23,42)


############################RETURN FONKSİYONLARI
##Fonksiyon çıktılarını girdi olarak kullanmak için..
def calc(varm,mois,charge):
    print((varm+mois)/charge)
calc(12,23,42)
#returnle yazalım...
def calc(varm,mois,charge):
    print((varm+mois)/charge)
calc(12,23,42)
#karmasıklastıralım...
def calc(varm,mois,charge):
    varm=varm*2
    mois=mois*4
    charge=charge/2
    output= (varm+mois/charge)
    return varm,mois,charge,output
calc(12,23,42) # 4 deger çıkıyor. cunku return 4 degeri birden yazdırdı.

##############################Fonksiyon içinden Fonksiyon Çağırma.
def calc(varm,mois,charge):
    varm=varm*2
    mois=mois*4
    charge=charge/2
    output= (varm+mois/charge)
    return varm,mois,charge,output

def calc2 (a,b):
    return a*5/b**2

#2 fonksiyonumuz oldu. calc ve calc2

def calc3 (varm: object, mois: object, charge: object, b: object) -> object:
    """

    :rtype: object
    """
    t=calc(varm,mois,charge)
    y=calc2(t,b)
    return(y*2)

    calc3(1,2,3,5)

#################Local ve Global Değişkenler

list1=[3,5]                  #global degiskenler
def func6 (a,b):
    c=a*b
    list1.append(c)           #local degiskenler
    print(list1)
    func6(4,8)

###########################QUİZ


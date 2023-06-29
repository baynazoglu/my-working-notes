##########################################NumPy
###neden numpy? hız ve vektorel oldugu ıcın hızlı
import numpy as np
a=[1,2,3,4,5]
b=[2,3,4,5,6]
#çarpıp sonucu yeni listeye eklemek old-schoolic way.
ab=[]
for i in range(0,len(a)):
    ab.append(a[i]*b[i])
    print(ab)

#numpy olarak;
import numpy as np
a=np.array([1,2,3,4,5])
b=np.array([2,3,4,5,6])
a*b
import seaborn as sns
### numpy arrayi olusturmak

np.array([1,2,3,4,5]) #bir liste.
np.zeros(5,dtype=int)
np.random.randint(0,10, size=5)
np.random.normal(10,4,(3,5)) #1. ort 10, stand sap,4 olan 3e 5lik array.


####Numpy Array Özellikleri...
#ndim:boyut sayısı
#shape: boyut bilgisi
#size and #dtype..

a=np.random.randint(10,size=5)
a
a.ndim #1 cıkacak boyutu 1 boyut.
a.shape #(5, ) tek boyut 5 eleman
a.size #5
a.dtype #int32


#######Reshaping..
b=np.random.randint(10,size=9)
b.reshape(3,3) #tekrar 3,3 reshapledik.
#sizeı 10 olsa hata verecekti 3e3e çeviremezdi.


####Index işlemleri..
b[0]
b[0:2] #slicing
b[0]= 213
c=b.reshape(3,3) #2 boyutlu array yapalım.
c
c[1,0] #(1,0 yani 1inci(aslında 2.) satır, 0.sütun.)
c[2,2]=6 # 2ye 2yi 6yla değiştirdk. ya float eklemek isteydik.
c[2,2]=7.5
# c 7 oldu int olarak devam.
c[:, 0] #tüm satırlar, ilk sütun.
c[1,:]
c[0:2,0:1]


#######Fancy Index
v=np.arange(0,30,3)#0dan 30a 3er 3er artarak oluştur.
v[4]
#bazen binlerce index seçmemiz gerekebilir...
indeks_cath=[1,2,3]
v[indeks_cath] #işte buna fancy index denir.


######Numpy Koşullu İşlemler...
# v nin içinde 10dan küçğk olan değerleri bul.
##old schoolic way..
ab=[]
for i in v:
    if i<10:
        ab.append(i)
    print(ab)

##with numpy
v[v < 10] #thats it.
v[v!= 3]

######Numpy Matematiksel İşlemler..
v /5
v * 5/10
v**2
v+3
np.subtract(v,2)
np.add(v,1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

#2 bilinmeyenli denklem...

## 5* x0 + x1 =12 ,   x0, 3*x1=10
a=np.array([[5,1],[1,3]]) #KATSAYILAR BIR ARRAY
b=np.array([12,10]) #SONUCLAR BIR ARRAY
np.linalg.solve(a,b)
###############################QUIZ
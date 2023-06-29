################VERİ GÖRSELLEŞTİRME
#####Matplot & Seaborn

#matplot : hepsinin atası. low-level. daha fazla çaba verirsin seaborna göre.

#kategorik değişken: sütun grafik. thats all. countplot bar.
#sayısal değişken: istatistiksel grafikler. histogram ve boxplot.

#python veri görselleştirme için çok da iyi degil. powerbi, tablue vs cok daha iyidir.

#############Kategorik Değişken Görselleştirmepip list --outdatedpip list --outdated

import matplotlib.pyplot as plt
matplotlib.pyplot
import seaborn as sns
matplotlib.colors
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns",None)
df=sns.load_dataset("titanic")
df.head()
#class who sex vs bunlar kategorik degiskenler.

df["sex"].value_counts()
#kategoriklerde value_counts cok elzem. grafiğe dökmek için.
df["sex"].value_counts().plot(kind="bar")
plt.show(block=True)  #block=true yazmayınca dondu.

##sayısal değişken görselleştirme.

#hist ve boxplot
#Hist
plt.hist(df["age"])
plt.show(block=True)

#Boxplot
plt.boxplot(df["fare"])
plt.show(block=True)

##Bu iki grafik istatistiksel 2 grafik. hem sayısal degiskenin aralık,frekansı hem de dagılım bilgisini verir. aykırı degere ulasabiliriz.

###########MatplotLib Özellikleri.

##plot
import numpy as np
x=np.array([1,8])
y=np.array([0,150])

plt.plot(x,y)
plt.show(block=True)
plt.plot(x,y, "o") #nokta koyacak
plt.show(block=True)

x=np.array([1,2,3,4,6,8])
y=np.array([0,1,2,3,4,6])
plt.plot(x,y) #nokta koyacak
plt.show(block=True)

#marker özelliği

y=np.array([0,12,23,3,44])
plt.plot(y,marker="o") #yuvarlark koyacak
plt.show(block=True)

###line özelliği

y=np.array([0,12,23,3,44])
plt.plot(y,linestyle="dashed",color="r") #dotted,dashdot vs
plt.show(block=True)

#birden fazla multiple lines..
y=np.array([0,12,23,3,44])
x=np.array([0,2,3,13,24])

plt.plot(x,linestyle="dashed",color="r") #dotted,dashdot vs
plt.show(block=True)
plt.plot(y) #dotted,dashdot vs
plt.show(block=True)

###Labels en mühimi..
y=np.array([0,12,23,3,44])
x=np.array([0,2,3,13,24])
plt.title("Ana Baslik")
plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.grid()
plt.plot(x)
plt.show(block=True)


##########3Subplots..

x=np.array([0,12,23,31,44,53,62,77])
y=np.array([0,2,3,13,24,28,29,32])
#plot1
plt.subplot(1,2,1) #1 satır 2 sütünlü grafiğin 1.grafiğini oluşturdum.
plt.title("1")
plt.plot(x)
#plt.show(block=True)
#plot2
plt.subplot(1,2,2) #1 satır 2 sütünlü grafiğin 1.grafiğini oluşturdum.
plt.title("2")
plt.plot(y)
plt.show(block=True)

########Seaborn ile veri görselleştirme...
#daha seri bir yoldur değişkenleri genellik seabornla görsellestirecegiz.

from matplotlib import pyplot as plt
#advanced way..
df=sns.load_dataset("tips")
df.head()
df["sex"].value_counts()
sns.countplot(x= df["sex"],data= df)
plt.show(block=True)

##sayısal değişkenler için:
sns.boxplot(x=df["total_bill"],data=df)
plt.show(block=True)

####pandas histogramı...
df["total_bill"].hist()
plt.show(block=True)

###################QUIZZZZZZZZZZZZ
df["total_bill"].plot.barh()
plt.show(block=True)


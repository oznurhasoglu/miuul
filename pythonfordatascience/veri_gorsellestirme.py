################################################
#Matplotlib ve Seaborn
#matplotlib lowlevel
#seaborn highlewel
################################################

#veri görselleştirme için veri tabanına bağlı BI(iş zekası araçları) ile görselleştirilmesi daha mantıklıdır.

# kategorik değişkeni: sütun grafiği/pasta grafiği(excel)  --> seaborn countplot bar
#sayısal değişkeni: histogram/boxplot --- ikisi de dağılım gösterir. boxpot aykırı değerleri de gösterir.

#################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df= sns.load_dataset("titanic")
df.head()
####################################################
#KATEGORİK DEĞİŞKENİ GÖRSELLEŞTİRME
df["sex"].value_counts().plot(kind = "bar")
plt.show()

###################################################
#SAYISAL DEĞİŞKENİ GÖRSELLEŞTİRME
plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

####################################################
#MATPLOTLİB ÖZELLİKLERİ
####################################################

################PLOT: VERİLEN NOKTALARI GÖRSELLEŞTİRİR
x= np.array([3,8])
y= np.array([0,150])

plt.plot(x,y) #iki nokta arasına bir çizgi çeker
plt.plot(x,y, "o") # o parametresiyle iki noktaya birer nokta koyar

#birden fazla nokta olduğunda
x= np.array([2,5,7,11])
y= np.array([0,3,6,10])

plt.plot(x,y)
plt.plot(x,y, "o")

####################### MARKER
z= np.array([2,5,7,11]) #bu nokalara bir işaret yani marker koymak istiyorum
plt.plot(z, marker= '*') #marker olarak seçebileceklerin:: "o,*,.,,,x,X,+,P,S,D,d,p,H,h"


####################### LİNE
plt.plot(z, linestyle= "dashdot", color="r") #dashed, dotted, dashdot
#plt.show() şu aan gereksiz bunu yazmak ama başka idelerde gerekli olacak elin alışsın


###################### LABELS
plt.title("BU GRAFİĞİN ANA BAŞLIĞI")
plt.xlabel("x ekseni adı")
plt.ylabel("y ekseni adı")
plt.grid() #arkayaa grid ekler


######################## SUBPLOTS: birden fazla görselin birlikte görselleştirilmesi
x= np.array([80,85,90,95,100,105,110,115,120])
y= np.array([240,250,260,270,280,290,300,310,320])

z= np.array([8,8,9,9,10,15,11,15,12])
t= np.array([24,20,26,27,280,29,30,31,32])

#plot1
plt.subplot(1,2,1) # 1 satırlık ve 2 sütunluk bir grafik oluşturuyorum. Bu 1. (,,1)
plt.title("2")
plt.plot(z,t)

#plot2
plt.subplot(1,2,2) # 2. grafiği ç,ziyorum
plt.title("1")
plt.plot(x,y)

##############################################################
#SEABORN ÖZELLİKLERİ
###########################################################
df= sns.load_dataset("tips")
df.head()

#kategorik değişken görselleştireme
df["sex"].value_counts()
sns.countplot(x= df["sex"], data = df)
#matplotta şöyleydi: df["sex"].value_counts().plot(kind = "bar")

#sayısal değişken görselleştirme
sns.boxplot(x= df["total_bill"])

df["total_bill"].hist()

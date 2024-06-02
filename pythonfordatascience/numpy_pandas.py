import numpy as np
import pandas as pd

#döngü yazmadan iki listeyi np arrayine çevirerek çarpabiliriz
a=np.array([1,2,3,4])
b=np.array([4,5,6,7])
a*b

#0'lardan oluşan bir array, kaç adet olacağı ve tipini de belirttik
np.zeros(10,int)
#random sınıfından randint metoduyla istediğimiz aralıkta (5 - 82) istediğimiz  kadar (size) rastgele sayı ürettirdik
np.random.randint(5,82,size=10) #eğer başlangıç değeri belirtmezsek default sıfırrdan o sayıya kadar alır.
#random sınıfının normal metoduyla ortalaması 10 olan standart sapması 4 olan 3*4 boyutunda normal dağılımda bir matris oluşturduk.
np.random.normal(10,4,(3,4))

"""numpy özellikleri"""
a= np.random.randint(10, size=5)
a.ndim #boyut bilgisi (kaç boyutlu)
a.shape #boyut bilgisi (satırxsütun)
a.size #eleman sayısı
a.dtype #tip bilgisi

"""reshaping"""
ar= np.random.randint(1,10, size=9)
ar.reshape(3,3) #9 elemanlı tek boyutlu arrayi 3x3 iki boyutlu hale getirdik


"""index işlemleri"""
ar= np.random.randint(10, size=10)
ar[0]
ar[0:5]
ar[0]=999

#çok boyutluysa
ar2= np.random.randint(10, size=(3,5))
ar2[0,0]
ar2[1,1]
ar2[2,3]=999
#float bir değer atamaya çalışırsam onun sadece int kısmını alır
ar2[:,0] #bütün satırları ve 0. sütunu seç
ar2[0:2,0:3] #0 ve 1. satırları, 0,1,2. sütunları getir


"""fancy index"""
v = np.arange(0,30,3)  #arange metodu 0dan 30a kadar 3er 3er artan bir array oluşturmaya yaradı
#bir arraydeki birden fazla indexse ulaşmak istediğimde fancy index kullanırım.
#yani index kısmına bir liste girerim o listede yer alan tüm indexleri getirir bana
catch=[1,3,5]
v[catch]  #1. 3. ve 5. indexi getiriyor


"""koşullu işlemler"""
v<15 #bu kod bana v arrayi içindeki 15ten küçük olan elemanları true, büyük ve eşit olanları false şeklinde bir listede döndürür.

v[v<15] #bu kod satırı ise true olanları getirir. fancy metoduna bağlı


"""matematiksel işlemler"""
#bir arraydeki tüm elemanlara aynı matematiksel işlemleri uygulamak için:
v/3 #tüm elemanları 3e böler
v**2-1 #tüm elemanların karesini alıp bir eksiltir

#operatörlerle işlem yapabildiğimiz gibi metotlar da vardır
np.subtract(v,2) #tüm elemanlarda iki çıkar
np.add(v,5)
np.mean(v) #tüm elemanların ortalamasını al
np.sum(v) #tüm elemanları topla
np.min(v)
np.max(v)
np.var(v) #varyansını hesapla

"""numpy ile iki bilinmeyenli denklem çözümü"""
#5x + y = 12
#x + 3y = 10 olsun. x ve y' yi bulmak için:
katsayilar= np.array([[5,1],[1,3]]) #xin katsayılarını ve ynin katsayılarını bir listede yazdık
sonuclar= np.array([12,10]) #sırasıyla sonuçları yazdık

np.linalg.solve(katsayilar,sonuclar) #linalg metoduyla çözüyoruz.

"""*********************************************************************************************************************"""
s = pd.Series([12,33,77,65,81])
s.index #indexler hakkında bilgi verir 0'dan başlar 5'e kadar devam eder 1'er artar diye bilgi verir.
s.dtype #seri içindeki verilerin tipi
s.size #serinin boyutu (eleman sayısı)
s.ndim #serinin kaç boyutlu olduğu
s.values # serideki değerleri ndarray olarask döndürür çünkü values dediğimiz zaman indexle işimiz olmadığını belirtmiş oluyoruz.
type(s.values) #ndarray
s.head(3) #serinin ilk 3 elemanı
s.tail(3) #serinin son 3 elemanı


"""veri okuma"""
df = pd.read_csv("datasets/advertising.csv")
df.head()
#csv json excel sql sas spss html pickle gibi birçok formattaki veriyi okuyabilir


"""veriye hızlı bakış"""
import seaborn as sns
df = sns.load_dataset("titanic") #seaborn içindeki hazır veri setlerinden titanic veri setini tanımladık.
df.head()
df.tail()
df.info() #sütular ve veriler hakkında bilgi
df.shape #kaç satır kaç sütun
df.columns #değişkenler (sütunları) döndürür
df.index
df.describe() #veri setinin istatistiklerine (sayısal değişkenlerin özetlerine) erişim
df.describe().T #transpozunu alarak daha okunabilir hale getirdim
df.isnull().values.any() # hiç null değer var mı bir tane bile olsa true döner
df.isnull().sum() #herbir değişkende kaç adet eksik veri var onu sayar

#kategorik değişkende kaç farklı kategori var ve bu kategoriler neler öğrenmek için:
df["sex"].value_counts()


"""pandasta seçim işlemleri"""
df = sns.load_dataset("titanic")
df[0:13]  #veri setinin ilk 13 satırını almak için
df.drop(0,axis=0)  #satırlardan 0. satırı sil ya da:
silinecekler= [0,1,3,5]
df.drop(silinecekler,axis=0)  # listede verilen tüm satırları sil. inplace=True argümanı eklersek bu değişikliği kalıcı yapar

"""değişkeni indexe çevirmek ya da tam tersi"""
df["age"].head()
df.index = df["age"]
df.drop("age", axis=1, inplace=True)

#tam tersi için de:
df["yeni sütun"] = df.index

#indexi silip kolon olarak eklemek için daha kısa başka bir yol
df.reset_index()

"""değişkenler üzerinde işlemler"""
#çok fazla değişken(kolon) olduğunda hepsini görmek istersek tsnımlarken şöyle yazıyoruz:
df = sns.load_dataset("titanic")
pd.set_option("display.max_columns", None)

"age" in df  #true/false döndürür
df["age"]  #bir kolonu ya böyle ya da aşağıdaki gibi seçeriz. AMA BU ŞEKİLDE SEÇERKEN DF OLARK DEĞİL SERİES OLARAK DÖNER.
df.age
type(df["age"])   #pandas.series
#DATAFRAME OLARAK DÖNMESİNİ İSTİYORSAK Bİ TANE DAHA [] KOYARIZ:
df[["age"]]
type(df[["age"]])  #dataframe

#birden fazla değişken seçmek için
df[["age", "alive", "adult_male"]]

#yeni bir değişken (kolon ekleme) yaş kolonunun karesini yrni bir kolon olarak ekleyelim
df["yeni kolon adı"] = df["age"]**2

#değişken sile
df.drop("yeni kolon adı", axis=1)

"""loc iloc"""
df.iloc[0:3]  #0 1 2. satırları seçer. 'e kadar mantığı
df.loc[0:3]   #0 1 2 3. satırları seçer. sonuncu dahil

#iloc integer temellidir. istediğimiz değişkenlerin indeksini yazmak zorundayız.
df.iloc[0:3,3]
#loc label temellidir. yani direkt istediğimiz değişkenlerin adını yazabiliriz.
df.loc[0:3, "age"]

col_names= ["age","sex","alive"]
df.loc[0:3, col_names]

"""koşullu işlemler"""
df = sns.load_dataset("titanic")
df[df["age"]>50].head() #yaş değişkeni 50den büyük olan verileri getirir
df[df["age"]>50].count() #bu da kaç tanesinin den büyük olduğunu sayar (tüm kolonlar için)
df[df["age"]>50]["age"].count() #bu da kaç tanesinin den büyük olduğunu sayar (sadece age kolonu için)
df.loc[df["age"]>50, ["age","class"]] #yaşı 50den büyük olanların age ve class sütunlarını getir

df_new = df.loc[(df["age"]>50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),["age","class", "embark_town"]]
df_new["embark_town"].value_counts()


"""toplulaştırma ve gruplama işlemleri fonksiyonları"""
df = sns.load_dataset("titanic")

df["age"].mean()  #yaşların ortalamsını alır
df.groupby("sex")["age"].mean() #yaşların cinsiyete göre ortalamsını alır
df.groupby("sex").agg({"age":"mean"}) #yaşların cinsiyete göre ortalamsını almanın tercih edilen yolu
df.groupby("sex").agg({"age": ["mean", "sum"]}) #yaşların cinsiyete göre hem ortalamsını hem toplamını alma
df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived":"mean"})  #cinsiyete göre yaş değişkeninin ortalamsı ve toplamı ve survived değişkeninin ortalaması
df.groupby(["sex","embark_town"]).agg({"age": "mean",
                       "survived":"mean"}) #birden fazla değişkene göre gruplama
df.groupby(["sex","embark_town", "class"]).agg({"age": "mean",
                       "survived":"mean"}) #birden fazla değişkene göre gruplama
df.groupby(["sex","embark_town","class"]).agg({
    "age": "mean",
    "survived":"mean",
    "sex": "count"}) #birden fazla değişkene göre gruplama


"""Pivot Table""" #veri setini kırılımlar açısından değerlendirme
import seaborn as sns
df = sns.load_dataset("titanic")
pd.set_option("display.max_columns", None) #çıktıda tüm satırları gösterir
pd.set_option("display.width", 500) #çıktıda gösterim genişliğini ayarlar
df.head()

df.pivot_table("survived", "sex", "embarked", aggfunc= "std")

#farklı boyutlar ekleyelim
df.pivot_table("survived", "sex", ["embarked","class"])

#sayısal değişkeni kategorik değişkene çevirmek cut/qcut fonksiyonları ile
df["new_age"] = pd.cut(df["age"],[0,10,18,25,40,90]) #neyi böleceğim,hangi aralıklarda böleceğim
df.head()

df.pivot_table("survived","sex","new_age")
df.pivot_table("survived","sex",["new_age","class"])

"""Apply ve Lambda""" #veri setini kırılımlar açısından değerlendirme
df = sns.load_dataset("titanic")
pd.set_option("display.max_columns", None) #çıktıda tüm satırları gösterir
pd.set_option("display.width", 500) #çıktıda gösterim genişliğini ayarlar

#apply bir fonksiyonu biröok saatır ya da sütuna uygulayabilmek için
#lambda kullan at fonksiyonlar yazmak için
#bir şeyi döngülerle yazmak yerine tek satırda kolayca yazmak için

#örnek için iki tane yeni kolon oluşturuyorum
df["age2"] = df["age"]*2
df["age3"] = df["age"]*3

#seçtiğim kolonlara aply aracılığıyla lambda ile yazdığım fonksiyonu uyguluyorum.
df[["age","age2","age3"]].apply(lambda x: x/10).head()

#kolonları tek tek seçmek yerine loc ile seçiyorum
df.loc[:,df.columns.str.contains("age")].apply(lambda x: x/10).head()

#farklı fonksiyonlar uygulayalım normalizasyon işlemi yapalım
df[["age","age2","age3"]].apply(lambda x: (x-x.mean())/(x.std())).head()

#ya da böyle de olur
def standart_scaler(x):
    return (x-x.mean())/x.std()
df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

#bunları yeni kolon olarak aatamak istersek yani yaptığımız değişikleri dfe kaydetmek için
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
df.head()

"""Birleştirme İşlemleri""" #concat ve merge
#yeni bir dataframe oluşturalım. rastgele bir np arrayi olulturup bunu dataframee çeviriyorum
m= np.random.randint(1,30,size=(5,3))
df1= pd.DataFrame(m, columns=["var1","var2","var3"])
df2= df1+99

pd.concat([df1,df2], ignore_index=True) #iki dfi bir listede verdik ve alt alta birleştirdi. indexleri yeniden sıralaması için
pd.concat([df1,df2], axis=1) #axisi değiştirirsek yan yana birleştirir

#merge ile örnekler
df1 = pd.DataFrame({"employees": ["mark","john","dennis","maria"],
                    "group": ["accounting","engineering","engineering","hr"]})

df2 = pd.DataFrame({"employees": ["mark","john","dennis","maria"],
                    "start_date": [2010,2009,2014,2019]})

df4 = pd.DataFrame({"group": ["accounting","engineering","hr"],
                    "manager": ["Caner","Mustafa","Berkcan"]})

df3 = pd.merge(df1,df2) #default olarak employeesa göre grupladı ama özellikle belirtmek istersek şöyle:
pd.merge(df1,df2, on="employees")

pd.merge(df3,df4)












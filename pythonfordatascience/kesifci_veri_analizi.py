### GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ ###

################################## 1. GENEL RESİM #########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df= sns.load_dataset("titanic")
df.head()
df.tail()
df.shape #verinin boyutu
df.dtypes  #kolonların tipini verir
df.info() #kolonlar, içindeki null olmayan değer sayıları ve veri tipleri
df.columns
df.index #index kaçta başlayıp kaçta bitiyor kaçar artıyor
df.isnull().values.any() #hiç null değer var mı
df.isnull().sum() #hangi kolonda kaç tane var
df.describe().T  #sayısal kolonların istatistiği

##################### şimdi bunları bir fonksiyonda birleştirelim ################
def check_df(dataframe,head=5):
    print("############### SHAPE ######################")
    print(dataframe.shape)
    print("############### INDEX ######################")
    print(dataframe.index)
    print("############### TYPES ######################")
    print(dataframe.dtypes)
    print("############### HEAD ######################")
    print(dataframe.head(head))
    print("############### TAIL ######################")
    print(dataframe.tail(head))
    print("############### NULL ######################")
    print(dataframe.isnull().sum())
    print("############### QUANTILES ######################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)

########################################## 2. KATEGORİK DEĞİŞKEN ANALİZİ ##################################
df["embarked"].value_counts()  #kategorik değişkende kaç ayrı kategori ve bu kategorilerde kaç değer olduğunu görüyoruz
df["sex"].unique() #kat. değişkendeki her bir kategoriyi gösterir
df["sex"].nunique() #kat. değişkendeki kaç tane kategori olduğunu gösterir

# her değişkene bu işlemleri tek tek uygulayamayacağımıza göre bunun için bi fonk. yazalım. cat_summary
# ama önce kategorik olan sınıfları tespit edecek bir fonksiyon lazım.  cat_cols
# tipi object, bool, category olan değişkenler zaten kategoriktir.
# ama sayısal gibi görünen int vb. tipte fakat sayısal olmayan değişkenleri de tespit etmeliyiz.

# kategorik değişkenler
cat_cols = [col for col in df.columns if df[col].dtypes in ["category", "object", "bool"]]
# sayısal gibi görünen kategorik değişkenler
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int32", "float32", "int64", "float64"]]
# kardinal değişkenler
cat_but_car = [col for col in df.columns if  df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat  # num but cat olanlar kategorik sayılacağı için ekledik
cat_cols = [col for col in cat_cols if col not in cat_but_car]  # son olarak varsa kardinal değişkenleri cat collar içinden çıkarıyoruz


#bi kontrol edelim cat colları doğru seçmiş miyiz? evet hepsinin 10dan az uniqe değeri var
df[cat_cols].nunique()

#sayısal değişkenlere bakalım
num_cols= [col for col in df.columns if col not in cat_cols]  #age ve fare

#bize categorik değişkenlerle ilgili kategori, veri sayısı ve kategorilerin yüzdelik oranını veren bir fonksiyon yazalım
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("********************************************")
    if plot:
        sns.countplot(x= dataframe[col_name], data= dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df,col, plot=True)

########################################## 3. SAYISAL DEĞİŞKEN ANALİZİ ##################################
df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64", "int32", "float32"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles= [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot= True)

for col in num_cols:
    num_summary(df, col, plot=True)


###################################################################################################
#ŞİMDİ HERHANGİ BİR VERİ SETİNDE KATEGORİK, NUMERİK, KARDİNAL DEĞİŞKENLERİ DÖNDÜRECEK FONKS YAZALIM
###################################################################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
  """
  VERİ SETİNDE KATEGORİK, NUMERİK, KARDİNAL DEĞİŞKENLERİ DÖNDÜRECEK FONKSİYON

  Parameters
  ----------
  dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
  cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
  car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

  Returns
  -------
  cat_cols: list
        kategorik değişken listesi
  num_cols: list
        numerik değişken listesi
  cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

  Notes
  -------
  cat_cols + num_cols + cat_but_car = toplam değişken sayısı
  num_but_cat, cat_cols'un içerisinde

  """
  # kategorik değişkenler
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["category", "object", "bool"]]
  # sayısal gibi görünen kategorik değişkenler
  num_but_cat = [col for col in dataframe.columns if
                 dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int32", "float32", "int64", "float64"]]
  # kardinal değişkenler
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]
  cat_cols = cat_cols + num_but_cat  # num but cat olanlar kategorik sayılacağı için ekledik
  cat_cols = [col for col in cat_cols if
              col not in cat_but_car]  # son olarak varsa kardinal değişkenleri cat collar içinden çıkarıyoruz

  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  print("Observation: ", dataframe.shape[0])
  print("Variables: ", dataframe.shape[1])
  print("cat_cols: ", len(cat_cols))
  print("num_cols: ", len(num_cols))
  print("cat_but_car: ", len(cat_but_car))
  print("num_but_cat: ", len(num_but_cat))

  return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

######################## 4. HEDEF DEĞİŞKEN ANALİZİ #########################
#bu veri setinde hedef değişken survived
df["survived"].value_counts()
cat_summary(df,"survived",plot=True)
#hedef değişkenin diğer değişkenlerle ilişkisine bakcaz

#kategorik değişkenlerle analizi
df.groupby("sex")["survived"].mean() #fonksiyonla yazıp tüm değişkenler için bakalım

def target_summary_with_cat(dataframe, target, categorical_col):
    print({"TARGET MEAN": pd.DataFrame(dataframe.groupby(categorical_col)[target].mean())})

for col in cat_cols:
    target_summary_with_cat(df,"survived", col)

#sayısal değişkenlerle analizi
#sayısal değişkenlerle analiz ederken groupbyda hedef değişkenin yerini değiştirdik dikkat
df.groupby("survived")["age"].mean()
#ya da daha güzel bir çıktı için agg kullan
df.groupby("survived").agg({"age":"mean"})

#fonksiyonla yazalım
def target_summary_with_num(dataframe,target,numerical_col):
    print(pd.DataFrame(df.groupby(target).agg({numerical_col:"mean"})), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"survived", col)

######################3 5. KORELASYON ANALİZİ #############################
df= pd.read_csv("datasets/breast_cancer.csv")
df.head()
#baştaki id ve sondaki unnamed değişkenlerini istemediğim için onları dahil etmeyeceğim
df = df.iloc[:,1:-1]

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#numerik kolonlar arasındaki korelasyonu buluyoruz.
corr = df[num_cols].corr()

#korelasyonu heatmap aracılığıyla görselleştiriyoruz. koyu mavi 1 yönündeki, kırmızı -1 yönündeki korelasyon.
sns.set(rc= {"figure.figsize": (12,12)})
sns.heatmap(corr, cmap="RdBu")
#plt.show()

#yüksek korelasyonlu değişkenlerin silinmesi
#negatif veya pozitif korelasyon fark etmediği için mutlak değerlerini alıyorum öncelikle

cor_matrix= df.corr(numeric_only= True).abs()

#oluşan matriste hem a değişkeni ile b değişkeninin korelasyonu, hem b ile a'nın korelasyonu görünüyor. yani gereksiz kalabalık bir matris.
#aynı çaprazlama olanların kaldırılması için aşağıdaki kodu yazıyoruz. köşegen ve altındaki değerleri nan ile dolduruoyr.
upper_triangle_matrix=  cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k= 1).astype(bool))

#oluşturduğumuz matriste 0.9'dan büyük korelasyonu olan değişkenleri seçip drop_list'e attık.
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.9)]

# 31 kolonluk datframeden bu değişkenleri silince 21 kolon kalıyor
df.drop(drop_list, axis= 1)

######################################
#şimdi tüm bunları fonksiyonlaştıralım
######################################
def high_correlated_cols(dataframe, cor_th= 0.9, plot=False):
    corr = dataframe.corr(numeric_only= True)
    cor_matrix = corr.abs() #negatif veya pozitif korelasyon fark etmediği için mutlak değerlerini alıyorum öncelikle
    upper_triangle_matrix=cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)) # oluşturduğumuz matriste 0.9'dan büyük korelasyonu olan değişkenleri seçip drop_list'e attık.
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > cor_th)] #oluşturduğumuz matriste 0.9'dan büyük korelasyonu olan değişkenleri seçip drop_list'e attık.
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list,axis = 1)

#sonuçta oluşan matrise de bir bakalım
high_correlated_cols(df.drop(drop_list,axis = 1), plot=True)


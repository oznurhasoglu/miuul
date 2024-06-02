################### KÜTÜPHANE TANIMLAMALARI #####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################### TERMİNAL DÜZENLEMELERİ #####################
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################### DATASET YÜKLENMESİ #####################
df= sns.load_dataset("titanic")

################### DATASET İNCELEMESİ #####################
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

################### BOOL DEĞİŞKENLERİN DÖNÜŞÜMÜ #####################
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

################### DEĞİŞKENLERİN ANALİZİ #####################
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

################### KATEGORİK DEĞİŞKENLER ÜZERİNDE İNCELEME #####################
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("********************************************")
    if plot:
        sns.countplot(x= dataframe[col_name], data= dataframe)
        plt.show(block=True)

for col in cat_cols:
        cat_summary(df,col, plot=True)

################### NUMERİK DEĞİŞKENLER ÜZERİNDE İNCELEME #####################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles= [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

################### HEDEF DEĞİŞKEN ANALİZİ ##################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print({"TARGET MEAN": pd.DataFrame(dataframe.groupby(categorical_col)[target].mean())})

for col in cat_cols:
    target_summary_with_cat(df,"survived", col)


def target_summary_with_num(dataframe,target,numerical_col):
    print(pd.DataFrame(df.groupby(target).agg({numerical_col:"mean"})), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"survived", col)

################### KORELASYON ANALİZİ ##################################
df= pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:,1:-1]

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









#############################################
# KURAL TABANLI SINIFLANDIRMA ILE POTANSIYEL MÜŞTERI GETIRISI HESAPLAMA
#############################################
""" İŞ PROBLEMİ
-Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.
Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor."""

""" VERİ SETİ HK.
Veri seti tekilleştirilmemiştir. Yani aynı demografik bilgilere sahip birden fazla satır olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı
"""


#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/persona.csv")
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

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()
#df.groupby("PRICE").agg({"PRICE":"count"})

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()

#df.groupby("COUNTRY")["PRICE"].count()
#df.groupby("COUNTRY")[["PRICE"]].count()
#df.groupby("COUNTRY")["COUNTRY"].count()
#df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})
df.pivot_table(values=["PRICE"],index=["COUNTRY"],aggfunc="sum")


# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()


# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby(['COUNTRY']).agg({"PRICE": "mean"})


# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(['SOURCE']).agg({"PRICE": "mean"})


# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()


#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.

agg_df.reset_index(inplace=True)
#agg_df = agg_df.reset_index()
agg_df.head()


#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

# AGE değişkeninin nerelerden bölüneceğini belirtelim:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# age'i bölelim:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

#Kesişimlerin gözlem sayısı
pd.crosstab(agg_df["AGE"],agg_df["age_cat"])


#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

# değişken isimleri:
agg_df.columns

# gözlem değerlerine nasıl erişiriz?
for row in agg_df.values:
    print(row)

# COUNTRY, SOURCE, SEX ve age_cat değişkenlerinin DEĞERLERİNİ yan yana koymak ve alt tireyle birleştirmek istiyoruz.
# Bunu list comprehension ile yapabiliriz.
# Yukarıdaki döngüdeki gözlem değerlerinin bize lazım olanlarını seçecek şekilde işlemi gerçekletirelim:

# yontem 1
[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

# yontem 2
[row["COUNTRY"].upper() + '_' + row["SOURCE"].upper() + '_' + row["SEX"].upper() + '_' + row["age_cat"].upper() for index, row in agg_df.iterrows()]

# yontem 3
agg_deneme=agg_df.drop(["AGE", "PRICE"], axis=1)
agg_deneme.head()

['_'.join(i).upper() for i in agg_deneme.values]
agg_deneme["customers_level_based"] =['_'.join(i).upper() for i in agg_deneme.values]
agg_deneme.head()

# yontem 4
agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(),axis=1)

# Veri setine ekleyelim:
agg_df["customers_level_based"] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(),axis=1)
agg_df.head()

# Gereksiz değişkenleri çıkaralım:
agg_df1 = agg_df[["customers_level_based", "PRICE"]]
agg_df1.head()


# Amacımıza bir adım daha yaklaştık.
# Burada ufak bir problem var. Bir çok aynı segment olacak.
# örneğin USA_ANDROID_MALE_0_18 segmentinden birçok sayıda olabilir.
# kontrol edelim:
agg_df1["customers_level_based"].value_counts()

# Bu sebeple segmentlere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz.
agg_df1 = agg_df1.groupby("customers_level_based").agg({"PRICE": "mean"})

# customers_level_based index'te yer almaktadır. Bunu değişkene çevirelim.
agg_df1.reset_index(inplace=True)
agg_df1.head()

# kontrol edelim. her bir persona'nın 1 tane olmasını bekleriz:
agg_df1["customers_level_based"].value_counts()
agg_df1.head()


#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

agg_df1["SEGMENT"]= pd.qcut(agg_df1["PRICE"], 4, labels=["D", "C", "B", "A"]) #küçükten büyüğe !!!
agg_df1.head(30)


agg_df1.groupby("SEGMENT").agg({"PRICE": ["mean"]})

#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user]


# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user2]
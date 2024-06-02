"""PYTHON PRACTICES"""
#alistirma 2
text = "The goal is to turn data into information, and information into insight."
#short way
text = text.upper().replace(",", " ").replace(".", " ")
text = text.split()

"""
#long way
text = text.upper()
for i in text:
    if i=="." or i == "," or i==".":
        text = text.replace(i, " ")
text= text.strip(" ")
text= text.split()
"""

#alistirma 3
lst = ['D', 'A', 'T', "A", "S", "C", "I", "E", "N", "C", "E"]
len(lst)
lst[10]
lst[0:4]
lst.remove(lst[8])
lst.append("New")
lst.insert(8,"N")

#alistirma 4
dict = {"Christian": ["America",18],
        "Daisy": ["England",12],
        "Antonio": ["Spain",22],
        "Dante": ["Italy",25]}
dict.keys()
dict.values()
dict["Daisy"][1]= 13
dict["Ahmet"] = ["Turkey", 24]
dict.pop("Antonio")


#alistirma 5
l=[2,13,18,93,22]
evenl=[]
oddl=[]
def evenodd(list):
    [evenl.append(i) if i%2==0 else oddl.append(i) for i in list]
    return evenl,oddl
evenodd(l)

#alistirma 6
students= ["ali","veli", "öznur","zeynep", "talat", "ece"]
for i,student in enumerate(students,1):
    if i<=3:
        print(f"Mühendislik Fakültesi {i}. öğrenci: {student}")
    else:
        print(f"Tıp Fakültesi {i-3}. öğrenci: {student}")

#alistirma 7
ders_kodu= ["CMP1005", "PSY1001", "SEN2204"]
kredi= [3,4,2,4]
kontenjan= [30,75,150,25]

zip= list(zip(ders_kodu, kredi, kontenjan))
for i in zip:
    print(f"Kredisi {i[1]} olan {i[0]} kodlu dersin kontenjanı {i[2]} kişidir.")


#alistirma 8
set1= set(["data", "python"])
set2= set(["data", "function", "qcut","lambda","python", "miuul"])

if set1.issuperset(set2):
    print(set1.intersection(set2))
else:
    print(set2.difference(set1))


"""LIST COMPREHENSIONS PRACTICES"""
#alistirma1
import seaborn as sns
df= sns.load_dataset("car_crashes")

["NUM_" + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns]


#alistirma2
[col.upper() + "_FLAG" if col.find("no") == -1 else col.upper() for col in df.columns]

#alistirma3
og_list= ["abbrev", "no_previous"]
new_col = []
[new_col.append(col) for col in df.columns if col not in og_list]

new_df= df[new_col]
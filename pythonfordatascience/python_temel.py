# write docstring
def summer(arg1, arg2):
    """
    sum of given numbers as total
    :return:
    :param arg1: can be int or float
    :param arg2: can be int or float
    :return: can be int or float as total
    """
    total = arg1 + arg2
    return total

"""**************************************************************************************************"""
# interview question
def alternating(string):
    """
    verilen stringin tek indexlerini küçük harf çift indexlerini büyük harf yapan fonksiyon
    :param string: given string for change
    :return: changing string
    """
    new_string = ""
    for i in range(0, len(string)):
        if i%2 == 0:
            new_string += string[i].upper()
        else:
            new_string += string[i].lower()
    return new_string

alternating("oznur")

"""**************************************************************************************************"""
# same interview question, using enumerate
def alternating_with_enu(string):
    """
        verilen stringin tek indexlerini küçük harf çift indexlerini büyük harf yapan fonksiyon
        :param string: given string for change
        :return: changing string
    """
    new_string=""
    for index, letter in enumerate(string):
        if index%2==0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    return new_string
alternating_with_enu("oznur")

"""**************************************************************************************************"""
# other interview question, using enumerate
def divide_students(students):
    """
    verilen öğrenci listesindekileri tek indexlileri bir grup çift indexlileri bir grup yap.
    :param students: students list
    :return: new list of groups
    """
    groups = [[], []] #tek ve çiftleri içerecek iki grup barındıran bir liste tanımladım.
    for index, student in enumerate(students):
        if index%2 == 0:
            groups[0].append(students[index])
        else:
            groups[1].append(students[index])
    return groups

students = ["öznur", "büşra", "gülizar", "osman", "latife"]
divide_students(students)

"""**************************************************************************************************"""
#list comprehensions
students= ["öznur","büşra","gülizar","osman", "latife"]
bad_students= ["osman", "latife"]

#make good students' name upper and bad students' name lower by using list comp.
[student.upper() if student not in bad_students else student.lower() for student in students]

"""**************************************************************************************************"""
#dict comprehensions
dictionary = {"a":1,
              "b":2,
              "c":3,
              "d":4}
dictionary.keys() #dict_keys(['a', 'b', 'c', 'd'])
dictionary.values() #dict_values([1, 2, 3, 4])
dictionary.items() #dict_items([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
dictionary["e"] = 5  #{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

{k: v ** 2 for (k, v) in dictionary.items()}
{k.upper(): v for (k, v) in dictionary.items()}
{k.upper(): v*2 for (k, v) in dictionary.items()}

"""**************************************************************************************************"""
#interview question
"""sayıların olduğu bir listem var. 
bu listeden çift sayıları alıp bir sözlüğe key olarak atayacak ve bu keylerin valueleri kareleri olacak."""
numbers = range(0,10)
dic={}
#long way
for n in numbers:
    if n % 2 == 0:
        dic[n]= n**2

#short way
{n: n**2 for n in numbers if n%2==0}

"""**************************************************************************************************"""
#interview question
"""key'i elimizdeki datasetin kolon isimleri (string), valuesi ise bu kolonlara uygulanmasını istediğimiz fonksiyonları
 içeren bir liste olan dictionary oluşturacağız. sadece sayısal değişkenler için yapacağız bunu."""

import seaborn as sns
df= sns.load_dataset("car_crashes")

null_dic = {}
agg_list = ["mean", "min", "max", "sum"]

num_cols= [col for col in df.columns if df[col].dtype != "O"] # "O" object veri tipini ifade ediyor.

#long way
for col in num_cols:
    null_dic[col] = agg_list

#short way
null_dic = {col: agg_list for col in num_cols}

#bu listedeki fonksiyonları kolonlardaki verilere uygulayalım. agg() metotu bunu yapar.
df[num_cols].agg(null_dic)
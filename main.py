import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from pandas import DataFrame
import operator


def zad1():
    ''' Wczytaj dane ze wszystkich plików do pojedynczej tablicy (używając Pandas). '''
    os.chdir(r'names')
    list_of_txt_files = glob.glob('*.txt')
    list_of_txt_files.sort()
    columns = []
    for name in list_of_txt_files:
        columns.append(str(name[3:7]) + " Name")
        columns.append(str(name[3:7]) + " Sex")
        columns.append(str(name[3:7]) + " Number")
    # print(columns)
    columns2 = ["Name", "Sex", "Number"]
    indexes = []
    for name_of_file in list_of_txt_files:
        df1 = pd.read_csv(name_of_file, sep=',', header=None)
        indexes.append(len(df1[0]))
    # print(max(indexes))

    # print(list_of_txt_files)
    df = pd.DataFrame(columns=columns, index=range(0, max(indexes)))


    index = 0
    for name_of_file in list_of_txt_files:
        df1 = pd.read_csv(name_of_file, sep=',', header=None)

        df[columns[index]] = df1[0]
        df[columns[index + 1]] = df1[1]
        df[columns[index + 2]] = df1[2]

        index = index + 3
        print(index)

    return df

def zad1_2():
    ''' Wczytaj dane ze wszystkich plików do pojedynczej tablicy (używając Pandas). '''
    os.chdir(r'names')
    list_of_txt_files = glob.glob('*.txt')
    list_of_txt_files.sort()
    columns = [ "Name", "Sex", "Number", "Year"]

    df = pd.DataFrame(columns=columns)

    for name_of_file in list_of_txt_files:
        print(name_of_file)
        df1 = pd.DataFrame(pd.read_csv(name_of_file, sep=',', header=None))
        df1 = df1.rename(columns={0:"Name", 1:"Sex", 2:"Number"})
        df1["Year"] = str(name_of_file[3:7])
        # print(type(df))
        df = df.append(df1)
    data2 = df.pivot(index=['Year', 'Name'], columns=['Sex'], values=['Number'])

    return data2
    # return df

def zad2(data):
    ''' Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie. '''
    print(data)
    unique_names = []
    list_of_years = list(range(1880, 2020))
    print(list_of_years)

    for column in data.columns:
        if column[5:9] == "Name":
            for name in data[column]:
                if name not in unique_names:
                    unique_names.append(name)
                    print(name)
    print(len(unique_names))
    # 99445

def zad3(data):
    ''' Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie rozróżniając imiona męskie i żeńskie. '''
    print(data)
    unique_female_names = []
    unique_male_names = []
    list_of_years = list(range(1880, 2020))

    for year in list_of_years:
        for index in data.index:
            # print(data.loc[index, str(year) + " Name"], data.loc[index, str(year) + " Sex"])

            if data.loc[index, str(year) + " Sex"] == 'F' and data.loc[index, str(year) + " Name"] not in unique_female_names:
                unique_female_names.append(data.loc[index, str(year) + " Name"])
                print(data.loc[index, str(year) + " Name"], data.loc[index, str(year) + " Sex"])

            if data.loc[index, str(year) + " Sex"] == 'M' and data.loc[index, str(year) + " Name"] not in unique_male_names:
                unique_male_names.append(data.loc[index, str(year) + " Name"])
                print(data.loc[index, str(year) + " Name"], data.loc[index, str(year) + " Sex"])
        # print()

    print(len(unique_female_names))
    # 68332
    print(len(unique_male_names))
    # 42054

def zad4(data):
    ''' Stwórz nowe kolumny frequency_male i frequency_female i określ popularność każdego z imion w danym każdym
    roku dzieląc liczbę razy, kiedy imię zostało nadane przez całkowita liczbę urodzeń dla danej płci.  '''
    list_of_years = list(range(1880, 2020))

    data2 = data.groupby(['1880 Sex']).sum()
    for year in list_of_years:
        data[str(year) + ' frequency_female'] = None
        data[str(year) + ' frequency_male'] = None
        sum_of_female_names = data2.loc['F', str(year) + ' Number']
        sum_of_male_names = data2.loc['M', str(year) + ' Number']
        print(str(year), sum_of_female_names)
        print(str(year), sum_of_male_names)

        data[str(year) + ' frequency_female'] = data[str(year) + ' Number']/sum_of_female_names
        data[str(year) + ' frequency_male'] = data[str(year) + ' Number']/sum_of_male_names

    print(data[['1880 Name', '1880 Sex', '1880 Number', '1880 frequency_female', '1880 frequency_male']])

    # 1880 90994.0
    # 1880 110490.0

def zad4_2(data):
    # print(data)
    data2 = data.groupby(level=0).sum()
    # print(data2)
    data_zad4 = data.copy()
    data_zad4['Number','frequency_female'] = data.loc[:,('Number', 'F')]/data2.loc[:,('Number', 'F')]
    data_zad4['Number', 'frequency_male'] = data.loc[:,('Number', 'M')]/data2.loc[:,('Number', 'M')]
    print(data_zad4)

def zad5_2(data):
    '''Określ i wyświetl wykres złożony z dwóch podwykresów, gdzie osią x jest skala czasu, a oś y reprezentuje:
    - liczbę urodzin w danym roku (wykres na górze)
    - stosunek liczby narodzin dziewczynek do liczby narodzin chłopców (wykres na dole)
     którym roku zanotowano najmniejszą, a w którym największą różnicę w liczbie urodzeń między chłopcami a dziewczynkami?
    '''

    list_of_years = list(range(1880, 2020))
    quantity_of_birth = []
    ratio_boys_to_girls = {}
    data2 = data.groupby(level=0).sum()
    # print(data2)

    for year in list_of_years:
        quantity_of_birth.append(data2.loc[str(year),('Number', 'F')] + data2.loc[str(year),('Number', 'M')])
        ratio_boys_to_girls[year] = (data2.loc[str(year),('Number', 'F')].astype(float) / data2.loc[str(year),('Number', 'M')].astype(float))

    print("Najwieksza roznica: ", max(ratio_boys_to_girls, key=ratio_boys_to_girls.get))
    print("Najmniejsza roznica: ", min(ratio_boys_to_girls, key=ratio_boys_to_girls.get))

    fig, ax = plt.subplots()
    ax.plot(list_of_years, quantity_of_birth, '--r')
    ax.plot(list_of_years, ratio_boys_to_girls.values(), '-b')
    plt.show()

def top_female(data):
    most_female_for_year = []
    list_of_years = list(range(1880, 2020))
    # print(data)

    female_sort_data = data.sort_values(by=['Year', ('Number', 'F')], ascending=False)
    # print(female_sort_data)
    # print(female_sort_data.loc[['1880'], [('Number', 'F')]].head(1000))
    top_all = {}
    for year in list_of_years:
        # print(year)
        most_female_for_year = []
        list1 = []
        top_year_list_dict = {}
        female_sort_data2 = female_sort_data.loc[[str(year)], [('Number', 'F')]].head(1000)
        # print(female_sort_data2.loc[(str(year)),('Number', 'F')])
        for idx in range(1000):
            most_female_for_year.append(female_sort_data2.index[idx][1])
            pass
        # print(female_sort_data2.index[0][1])
        # print(type(female_sort_data2.index[0]))
        # print(most_female_for_year)
        # print(type(female_sort_data2.loc[('1880'),('Number', 'F')]))

        list1 = female_sort_data2.loc[(str(year)), ('Number', 'F')].tolist()
        # print(list1)
        top_year_list_dict = dict(zip(most_female_for_year, list1))
        # print(top_year_list_dict)

        for key, value in top_year_list_dict.items():
            if key in top_all.keys():
                top_all[key] = top_all[key] + value
            else:
                top_all[key] = value

    # print(top_all)
    # print(len(top_all))
    sort_female = sorted(top_all.items(), key=lambda x: x[1], reverse=True)
    # print(len(sort_female))
    # print(type(sort_female))
    # print("Top female names:")
    # print(sort_female[:1000])
    return sort_female[:1000]

def top_male(data):
    most_male_for_year = []
    list_of_years = list(range(1880, 2020))

    male_sort_data = data.sort_values(by=['Year', ('Number', 'M')], ascending=False)

    top_all = {}
    for year in list_of_years:
        # print(year)
        most_male_for_year = []
        list1 = []
        top_year_list_dict = {}
        male_sort_data2 = male_sort_data.loc[[str(year)], [('Number', 'M')]].head(1000)

        for idx in range(1000):
            most_male_for_year.append(male_sort_data2.index[idx][1])

        list1 = male_sort_data2.loc[(str(year)), ('Number', 'M')].tolist()
        top_year_list_dict = dict(zip(most_male_for_year, list1))

        for key, value in top_year_list_dict.items():
            if key in top_all.keys():
                top_all[key] = top_all[key] + value
            else:
                top_all[key] = value

    sort_male = sorted(top_all.items(), key=lambda x: x[1], reverse=True)

    # print("Top male names:")
    # print(sort_male[:1000])
    return sort_male[:1000]

def zad6_2(data):
    '''Wyznacz 1000 najpopularniejszych imion dla każdej płci w całym zakresie czasowym,
     metoda powinna polegać na wyznaczeniu 1000 najpopularniejszych imion dla każdego roku i dla każdej płci a
     następnie ich zsumowaniu w celu ustalenia rankingu top 1000 dla każdej płci.'''
    print("Top female names:")
    print(top_female(data))
    # ('Mary', 4128052)
    print("Top male names:")
    print(top_male(data))
    # ('James', 5177716)

def zad7_2(data):
    '''
    Wyświetl wykresy zmian dla imion Harry i Marilin oraz pierwszego imienia w żeńskiego i męskiego w rankingu:
     - na osi Y po lewej liczbę razy kiedy imę zostało nadane w każdym roku (zanotuj ile razy nadano to imię w 1940, 1980 i 2019r)?
     - na osi Y po prawej popularność tych imion w każdym z lat
    '''
    # 'Mary', 'James', 'Harry', 'Marilin'
    mary_dict = {}
    james_dict = {}
    harry_dict = {}
    marilin_dict = {}

    # print(data)
    list_of_years = list(range(1880, 2020))
    special_years = [1940, 1980, 2019]

    special_mary_dict = {}
    special_james_dict = {}
    special_harry_dict = {}
    special_marilin_dict = {}

    # print(data.loc[('1880', 'Mary'), ('Number', 'F')])

    for year in list_of_years:
        try:
            mary_dict[year] = data.loc[(str(year), 'Mary'), ('Number', 'F')]
        except KeyError:
            mary_dict[year] = 0
        try:
            marilin_dict[year] = data.loc[(str(year), 'Marilin'), ('Number', 'F')]
        except KeyError:
            marilin_dict[year] = 0
        try:
            james_dict[year] = data.loc[(str(year), 'James'), ('Number', 'M')]
        except KeyError:
            james_dict[year] = 0
        try:
            harry_dict[year] = data.loc[(str(year), 'Harry'), ('Number', 'M')]
        except KeyError:
            harry_dict[year] = 0
        #--------------------------------------
        if year in special_years:
            try:
                special_mary_dict[year] = data.loc[(str(year), 'Mary'), ('Number', 'F')]
            except KeyError:
                special_mary_dict[year] = 0
            try:
                special_marilin_dict[year] = data.loc[(str(year), 'Marilin'), ('Number', 'F')]
            except KeyError:
                special_marilin_dict[year] = 0
            try:
                special_james_dict[year] = data.loc[(str(year), 'James'), ('Number', 'M')]
            except KeyError:
                special_james_dict[year] = 0
            try:
                special_harry_dict[year] = data.loc[(str(year), 'Harry'), ('Number', 'M')]
            except KeyError:
                special_harry_dict[year] = 0

    # print(mary_dict)
    # print(marilin_dict)
    # print(james_dict)
    # print(harry_dict)
    #
    # print(special_mary_dict)
    # print(special_marilin_dict)
    # print(special_james_dict)
    # print(special_harry_dict)

    fig, ax1 = plt.subplots()
    ax1.plot(special_years, special_mary_dict.values(), '--r')
    ax1.plot(special_years, special_marilin_dict.values(), '--b')
    ax1.plot(special_years, special_james_dict.values(), '--k')
    ax1.plot(special_years, special_harry_dict.values(), '--g')

    ax2 = ax1.twinx()
    ax2.plot(list_of_years, mary_dict.values(), '-r')
    ax2.plot(list_of_years, marilin_dict.values(), '-b')
    ax2.plot(list_of_years, james_dict.values(), '-k')
    ax2.plot(list_of_years, harry_dict.values(), '-g')
    ax2.legend(['Mary', 'Marilin', 'James', 'Harry'])
    plt.show()

def zad8_2(data):
    '''
    Wykreśl wykres z podziałem na lata i płeć zawierający informację jaki procent w danym roku
    stanowiły imiona należące do rankingu top1000. Wykres ten opisuje różnorodność imion,
    zanotuj rok w którym zaobserwowano największą różnicę w różnorodności między imionami męskimi a żeńskimi.
    '''
    # print(data)
    list_of_years = list(range(1880, 2020))
    top_female_ = top_female(data)
    top_male_ = top_male(data)
    # print(type(top_female_))
    # print(top_female_)
    # print(top_male_)

    sum_of_female_names_from_top1000 = 0
    dict_female_top1000 = {}
    sum_of_male_names_from_top1000 = 0
    dict_male_top1000 = {}


    dict_female_all = {}
    dict_male_all = {}
    data2 = data.groupby(level=0).sum()
    # print(data2)
    for year in list_of_years:
        dict_female_all[year] = data2.loc[str(year),('Number', 'F')]
        dict_male_all[year] = data2.loc[str(year),('Number', 'M')]
        # print(data2.loc[str(year),('Number', 'F')])
    # print(dict_female_all)


    for year in list_of_years:
        # print("YEAR", year)
        for name, q in top_female_:
            try:
                # print(name)
                # print(data.loc[(str(year), name), ('Number', 'F')])
                if pd.isna(data.loc[(str(year), name), ('Number', 'F')]):
                    pass
                else:
                    sum_of_female_names_from_top1000 = sum_of_female_names_from_top1000 + data.loc[(str(year), name), ('Number', 'F')]
                    # print(sum_of_female_names_from_top1000)
            except KeyError:
                pass
        # -----------------------
        for name, q in top_male_:
            try:
                if pd.isna(data.loc[(str(year), name), ('Number', 'M')]):
                    pass
                else:
                    sum_of_male_names_from_top1000 = sum_of_male_names_from_top1000 + data.loc[(str(year), name), ('Number', 'M')]
            except KeyError:
                pass

        # print("YEAR SUM",sum_of_female_names_from_top1000)
        dict_female_top1000[year] = sum_of_female_names_from_top1000
        dict_male_top1000[year] = sum_of_male_names_from_top1000
        sum_of_female_names_from_top1000 = 0
        sum_of_male_names_from_top1000 = 0
    # print(dict_female_top1000)
    # print(dict_female_all)

    female_ratio = []
    male_ratio = []
    for year in list_of_years:
        female_ratio.append(dict_female_top1000[year] / dict_female_all[year])
        male_ratio.append(dict_male_top1000[year] / dict_male_all[year])

    fig, ax = plt.subplots()
    ax.plot(list_of_years, female_ratio, '-r')
    ax.plot(list_of_years, male_ratio, '-b')
    plt.show()
    pass



def main():
    # data = zad1()
    data2 = zad1_2()
    # zad2(data)
    # zad3(data)
    # zad4(data)
    # zad4_2(data2)
    # zad5_2(data2)
    # zad6_2(data2)
    # zad7_2(data2)
    zad8_2(data2)



if __name__ == '__main__':
    main()











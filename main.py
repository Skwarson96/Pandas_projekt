import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import sqlite3
from pandas import DataFrame
import operator
import math

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
    # !!!!!!!!!!!!!!!
    #     print(data.groupby(level=1).sum())
    #

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


        # top_year_list_dict = {k: top_year_list_dict[k] for k in top_year_list_dict if not math.isnan(top_year_list_dict[k])}


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

    popularity_mary_dict = {}
    popularity_james_dict = {}
    popularity_harry_dict = {}
    popularity_marilin_dict = {}

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

    # -----------------------------------------------------------------
    # POPULARNOSC

    data2 = data.groupby(level=0).sum()
    # print(data2)


    for year in list_of_years:
        try:
            popularity_mary_dict[year] = data.loc[(str(year), 'Mary'), ('Number', 'F')] / data2.loc[str(year), ('Number', 'F')] * 100
        except KeyError:
            popularity_mary_dict[year] = 0
        try:
            popularity_marilin_dict[year] = data.loc[(str(year), 'Marilin'), ('Number', 'F')] / data2.loc[str(year), ('Number', 'F')] * 100
        except KeyError:
            popularity_marilin_dict[year] = 0
        try:
            popularity_james_dict[year] = data.loc[(str(year), 'James'), ('Number', 'M')] / data2.loc[str(year), ('Number', 'M')] * 100
        except KeyError:
            popularity_james_dict[year] = 0
        try:
            popularity_harry_dict[year] = data.loc[(str(year), 'Harry'), ('Number', 'M')] / data2.loc[str(year), ('Number', 'M')] * 100
        except KeyError:
            popularity_harry_dict[year] = 0



    # print(mary_dict)
    # print(marilin_dict)
    # print(james_dict)
    # print(harry_dict)
    #
    print('special_mary_dict', special_mary_dict)
    print('special_marilin_dict', special_marilin_dict)
    print('special_james_dict', special_james_dict)
    print('special_harry_dict', special_harry_dict)

    fig, ax1 = plt.subplots()
    ax1.plot(list_of_years, mary_dict.values(), '-r')
    ax1.plot(list_of_years, marilin_dict.values(), '-b')
    ax1.plot(list_of_years, james_dict.values(), '-k')
    ax1.plot(list_of_years, harry_dict.values(), '-g')
    ax1.legend(['Mary', 'Marilin', 'James', 'Harry'], loc='upper left')
    # print(special_mary_dict.keys())
    # print(type(special_mary_dict.keys()))
    ax1.plot(list(special_mary_dict.keys()), list(special_mary_dict.values()), 'or')
    ax1.plot(list(special_marilin_dict.keys()), list(special_marilin_dict.values()), 'ob')
    ax1.plot(list(special_james_dict.keys()), list(special_james_dict.values()), 'ok')
    ax1.plot(list(special_harry_dict.keys()), list(special_harry_dict.values()), 'og')

    ax2 = ax1.twinx()
    ax2.plot(list_of_years, popularity_mary_dict.values(), '--r')
    ax2.plot(list_of_years, popularity_marilin_dict.values(), '--b')
    ax2.plot(list_of_years, popularity_james_dict.values(), '--k')
    ax2.plot(list_of_years, popularity_harry_dict.values(), '--g')
    ax2.legend(['Mary%', 'Marilin%', 'James%', 'Harry%'], loc='upper right')


    ax1.set_title("Zadanie 7")
    ax1.set_xlabel("Rok")
    ax1.set_ylabel("Liczba nadanych imion")
    ax2.set_ylabel("Popularność imienia [%]")

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
    print('top_female_', top_female_)
    print('top_male_', top_male_)

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
        # if year == 1880:
        #     print("YEAR", year)
        for name, q in top_female_:
            try:
                # if year == 1880:
                #     print(name)
                #     print(data.loc[(str(year), name), ('Number', 'F')])
                if pd.isna(data.loc[(str(year), name), ('Number', 'F')]):
                    pass
                else:
                    sum_of_female_names_from_top1000 = sum_of_female_names_from_top1000 + data.loc[(str(year), name), ('Number', 'F')]
                    # if year == 1880:
                    #     print(sum_of_female_names_from_top1000)
            except KeyError:
                pass

        dict_female_top1000[year] = sum_of_female_names_from_top1000
        sum_of_female_names_from_top1000 = 0
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
        dict_male_top1000[year] = sum_of_male_names_from_top1000
        sum_of_male_names_from_top1000 = 0


    # print('dict_female_top1000',dict_female_top1000)
    # print('dict_female_all', dict_female_all)
    # print('dict_male_top1000', dict_male_top1000)
    # print('dict_male_all', dict_male_all)

    female_ratio = []
    male_ratio = []
    difference = {}

    for year in list_of_years:
        female_ratio.append(dict_female_top1000[year] / dict_female_all[year] * 100)
        male_ratio.append(dict_male_top1000[year] / dict_male_all[year] * 100)
        difference[year] = abs((dict_female_top1000[year] / dict_female_all[year]) - (dict_male_top1000[year] / dict_male_all[year]))

    # print(difference)
    print("Rok z najwieksza roznica:", max(difference, key=difference.get))


    fig, ax = plt.subplots()
    ax.plot(list_of_years, female_ratio, '-r')
    ax.plot(list_of_years, male_ratio, '-b')

    ax.legend(['female_ratio [%]', 'male_ratio [%]'], loc='upper right')
    plt.show()
    pass

def zad9_2(data):
    '''
    Zweryfikuj hipotezę czy prawdą jest, że w obserwowanym okresie rozkład ostatnich liter imion męskich uległ istotnej
    zmianie? W tym celu:
     - dokonaj agregacji wszystkich urodzeń w pełnym zbiorze danych z podziałem na rok i płeć i ostatnią literę,
     - wyodrębnij dane dla lat 1910, 1960, 2015
     - znormalizuj dane względem całkowitej liczby urodzin w danym roku
     - wyświetl dane popularności litery dla każdej płci w postaci wykresu słupkowego zawierającego
      poszczególne lata i gdzie słupki grupowane są wg litery. Zanotuj, dla której litery wystąpił
      największy wzrost/spadek między rokiem 1910 a 2015)
     - Dla 3 liter dla których zaobserwowano największą zmianę wyświetl przebieg trendu popularności w czasie
    '''

    # print("???")
    print(data)
    list_of_years = list(range(1880, 2020))
    special_years = ['1910', '1960', '2015']

    # data2 = data.unstack('Name')
    data2 = data.reset_index()
    print(data2)

    data2['Name'] = data2['Name'].str.strip().str[-1]
    print(data2)

    # print(data2.groupby(by=["Year", "Name"]).sum())
    data2 = data2.groupby(by=["Year", "Name"]).sum()
    print(data2)
    del data2['Year']
    del data2['Name']
    print(data2)
    # print(special_years[0])
    data_1910 = data2.loc[special_years[0]]
    data_1960 = data2.loc[special_years[1]]
    data_2015 = data2.loc[special_years[2]]
    print('data_1910', data_1910)
    data_1910_F_sum = data_1910.loc[:, ("Number",'F')].sum()
    data_1910_M_sum = data_1910.loc[:, ("Number",'M')].sum()
    # print('data_1910_F_sum', data_1910_F_sum)
    # print('data_1910_M_sum', data_1910_M_sum)
    data_1910[("Number", "F")] = data_1910[("Number", "F")]/ data_1910_F_sum
    data_1910[("Number", "M")] = data_1910[("Number", "M")] / data_1910_M_sum

    data_1960_F_sum = data_1960.loc[:, ("Number",'F')].sum()
    data_1960_M_sum = data_1960.loc[:, ("Number",'M')].sum()
    data_1960[("Number", "F")] = data_1960[("Number", "F")]/ data_1960_F_sum
    data_1960[("Number", "M")] = data_1960[("Number", "M")] / data_1960_M_sum

    data_2015_F_sum = data_2015.loc[:, ("Number",'F')].sum()
    data_2015_M_sum = data_2015.loc[:, ("Number",'M')].sum()
    data_2015[("Number", "F")] = data_2015[("Number", "F")]/ data_2015_F_sum
    data_2015[("Number", "M")] = data_2015[("Number", "M")] / data_2015_M_sum

    print('data_1910', data_1910)
    # print('sum data 1910 M', data_1910.loc[:, ("Number",'M')].sum())
    # print(data_1960)
    # print(data_2015)

    data_1910.loc['q', [("Number", "F"), ("Number", "M")]] = (0, 0)
    data_1910.loc['j', [("Number", "F"), ("Number", "M")]] = (0, 0)
    data_1910.sort_index(inplace=True)
    # labels = list(data_1910.loc[:,("Number", "F")].index.values)
    # print('labels', labels)
    # print(len(labels))

    data_1960.loc['q', [("Number", "F"), ("Number", "M")]] = (0, 0)
    data_1960.loc['j', [("Number", "F"), ("Number", "M")]] = (0, 0)
    data_1960.sort_index(inplace=True)
    # print(data_1960)

    # labels = list(data_1960.loc[:,("Number", "F")].index.values)
    # print('labels', labels)
    # print(len(labels))

    labels = list(data_2015.loc[:,("Number", "F")].index.values)
    # print('labels', labels)
    # print(len(labels))


    # print('data_1910.loc[:,("Number", "F")]', data_1910.loc[:,("Number", "F")])
    # print('data_1910.loc[:,("Number", "M")]', data_1910.loc[:,("Number", "M")])
    # print(type(data_1910.loc[:,("Number", "M")]))
    # print(data_1910.loc[:,("Number", "M")].values)
    # print(data_1910.loc[:,("Number", "M")].values*100)
    # F_list = list(data_1910.loc[:,("Number", "F")].values *100)
    # M_list = list(data_1910.loc[:,("Number", "M")].values *100)

    # print(F_list)
    # print('len F list', len(F_list))
    # print('len labels', len(labels))


    fig, ax = plt.subplots()
    x = np.arange(len(labels))

    barWidth = 0.1
    r1 = np.arange(len(labels))
    print(r1)
    r2 = [a + barWidth for a in r1]
    print(r2)
    r3 = [a + barWidth for a in r2]
    print(r3)
    r4 = [a + barWidth for a in r3]
    print(r4)
    r5 = [a + barWidth for a in r4]
    print(r5)
    r6 = [a + barWidth for a in r5]
    print(r6)

    ax.bar(r1, data_1910.loc[:,("Number", "F")].values *100, barWidth, label='1910 F')
    ax.bar(r2, data_1910.loc[:,("Number", "M")].values *100, barWidth, label='1910 M')

    ax.bar(r3, data_1960.loc[:,("Number", "F")].values *100, barWidth, label='1960 F')
    ax.bar(r4, data_1960.loc[:,("Number", "M")].values *100, barWidth, label='1960 M')

    ax.bar(r5, data_2015.loc[:,("Number", "F")].values *100, barWidth, label='2015 F')
    ax.bar(r6, data_2015.loc[:,("Number", "M")].values *100, barWidth, label='2015 M')

    ax.set_title('Zadanie 9')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    plt.xticks([r + barWidth for r in range(len(labels))], labels)
    ax.set_ylabel('%')
    ax.legend()

    plt.show()

def zad10_2(data):
    '''
    Znajdź imiona, które nadawane były zarówno dziewczynkom jak i chłopcom
    (zanotuj najpopularniejsze imię męskie i żeńskie)
    '''

    data2 = data.dropna()
    # print(data2)
    data2 = data2.sum(axis=1)
    # print(data2)
    data2 = data2.groupby(level="Name").sum()
    # print(data2)
    data2 = data2.sort_values()
    # print(data2)
    print(data2.index[-1])
    # print(type(data2))

def zad11_2(data):
    '''
    Spróbuj znaleźć najpopularniejsze imiona, które przez pewien czas były imionami żeńskimi/męskimi a następnie stały
     się imionami męskimi/żeńskimi.
     - możesz spróbować wyliczyć dla każdego imienia stosunek w którym nadawane było chłopcom i dziewczynkom
     - następnie oblicz zagregowaną wartość tego współczynnika w latach 1880-1920 oraz w okresie 2000-2020
      i na tej podstawie wybrać imiona dla których zaobserwowana
      zmiana jest największa (zanotuj dwa najpopularniejsze imiona)
     - wkreśl przebieg trendu dla tych imion
    '''

    print(data)

    data2 = data.dropna()
    print(data2)

def zad12():
    '''
    Wczytaj dane z bazy opisującej śmiertelność w okresie od 1959-2018r w poszczególnych grupach wiekowych: USA_ltper_1x1.sqlite,
    opis: https://www.mortality.org/Public/ExplanatoryNotes.php.
    Spróbuj zagregować dane już na etapie zapytania SQL.
    '''
    os.chdir('..')  # back to main folder
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")  # połączenie do bazy danych - pliku
    c = conn.cursor()

    cursor = conn.execute('SELECT * from USA_fltper_1x1')
    column_names = list(map(lambda x: x[0], cursor.description))
    # print(column_names)

    # data = pd.DataFrame(columns=column_names)
    data = pd.DataFrame()
    female_df = pd.read_sql_query('SELECT * FROM USA_fltper_1x1', conn)
    male_df = pd.read_sql_query('SELECT * FROM USA_mltper_1x1', conn)

    data = pd.concat([female_df, male_df], axis=0)
    # print(data)

    conn.close()


    # print(data)
    return data

def zad13(birth_data, death_data):
    '''
    Wyznacz przyrost naturalny w analizowanym okresie
    '''
    # przyrost naturalny = liczba urodzen - liczba zgonow
    # stopa przyrostu naturalnego = (liczba urodzen - liczba zgonow)/liczba mieszkancow

    # analizowany okres = 1959-2017

    # print(death_data)
    dff = death_data.groupby(["Year"]).dx.sum().reset_index()
    # dff = death_data.groupby(["Year"]).dx.mean()
    # print(dff)


    years = []
    years = list(dff['Year'])
    years = [str(int) for int in years]
    # print(years)

    # suma urodzen dla kazdej plci w kazdym roku
    # print(birth_data)
    birth_data_sum = birth_data.groupby(["Year"]).sum().reset_index()
    # print(birth_data_sum)

    # zostawienie interesujących lat 1959-2017
    birth_data_sum = birth_data_sum[birth_data_sum["Year"].isin(years)]
    birth_data_sum = birth_data_sum.reset_index()
    # print(birth_data_sum)

    birth_data_sum = birth_data_sum.sum(axis = 1, skipna = True)
    # print(birth_data_sum)
    # birth_data_sum["Sum"] = birth_data_sum[('Number', 'F')] + birth_data_sum[('Number', 'M')]
    # print(birth_data_sum)

    # print(dff)

    birthrate = {}

    for idx , year in enumerate(years):
        print(year, birth_data_sum[idx], dff['dx'][idx])
        birthrate[year] = int(birth_data_sum[idx]-dff['dx'][idx])
        pass

    # print(birthrate)
    fig, ax = plt.subplots()
    ax.plot(birthrate.keys(), birthrate.values(), '-r')
    # ax.set_xticklabels([1959, 2017, step=10])
    ax.set_title("Zadanie 13")
    plt.show()

def zad14(birth_data, death_data):
    '''
    Wyznacz i wyświetl współczynnik przeżywalności dzieci w pierwszym roku życia
    '''
    # print(birth_data)
    # print(death_data)

    dff = death_data.groupby(["Year"]).dx.sum().reset_index()
    # print(dff)
    years = []
    years = list(dff['Year'])
    years = [str(int) for int in years]

    birth_data_sum = birth_data.groupby(["Year"]).sum().reset_index()
    # print(birth_data_sum)


    birth_data_sum = birth_data_sum[birth_data_sum["Year"].isin(years)]
    # print(birth_data_sum)
    birth_data_sum = birth_data_sum.reset_index()
    # print(birth_data_sum)

    birth_data_sum.pop("index")
    # print(birth_data_sum)
    # birth_data_sum = birth_data_sum.pivot(birth_data_sum, values=[('Number','F'), ('Number','M')], rows=['Year'], cols=['Number'], aggfunc=np.sum, margins=True)
    birth_data_sum['Sum'] = birth_data_sum[('Number','F')] + birth_data_sum[('Number','M')]
    # print(birth_data_sum)

    death_data_for_0 = death_data[death_data.Age == 0].reset_index()
    death_data_for_0.pop("index")
    # print(death_data_for_0)

    # print(death_data_for_0[['Year', 'Sex', 'dx']])

    death_data_sum = death_data_for_0.groupby(['Year']).sum().reset_index()
    # print(death_data_sum)
    birth_data_sum['D'] = death_data_sum['dx']

    birth_data_sum['przezywalnosc'] = (birth_data_sum['Sum'] - birth_data_sum['D']) / birth_data_sum['Sum'] * 100
    # print(birth_data_sum)

    zad15_data = zad15(birth_data_sum, death_data)
    # print(zad15_data)
    zad15_data[2014] = None
    zad15_data[2015] = None
    zad15_data[2016] = None
    zad15_data[2017] = None

    birth_data_sum['D5'] = zad15_data.values()
    # print(birth_data_sum)
    birth_data_sum['przezywalnosc_5l'] = (birth_data_sum['Sum'] - birth_data_sum['D5']) / birth_data_sum['Sum'] * 100


    fig, ax = plt.subplots()
    ax.plot(birth_data_sum['przezywalnosc'], '-r')
    ax.plot(birth_data_sum['przezywalnosc_5l'], '-b')
    # ax.legend()
    plt.grid(True)
    ax.set_title("Zadanie 14, 15")
    plt.show()


def zad15(birth_data, death_data):
    '''
    Na wykresie z pkt 14 wyznacz współczynnik przeżywalności dzieci w pierwszych 5 latach życia
    (pamiętaj, że dla roku urodzenia x należy uwzględnić śmiertelność
     w grupie wiekowej 0 lat w roku x, 1rok w roku x+1 itd).
    '''

    dff = death_data.groupby(["Year"]).dx.sum().reset_index()
    years = []
    years = list(dff['Year'])

    # print(death_data)

    data_dict = {}

    data = death_data[['Sex', 'Year', 'Age', 'dx']]
    # print(data)
    data = data[(data.Age == 0) | (data.Age == 1) | (data.Age == 2) | (data.Age == 3) | (data.Age == 4)]
    # print(data)
    data = data.groupby(by=["Year", "Age"]).sum()
    # print(data)

    for year in years:
        try:
            data_dict[year] = int(data.loc[year, 0].values[0]) + \
                              int(data.loc[year + 1, 1].values[0]) + \
                              int(data.loc[year + 2, 2].values[0]) + \
                              int(data.loc[year + 3, 3].values[0]) + \
                              int(data.loc[year + 4, 4].values[0])
        except KeyError:
            pass

    # print(data_dict)

    return data_dict





def main():
    # data = zad1()
    birth_data = zad1_2()
    # zad2(data)
    # zad3(data)
    # zad4(data)
    # zad4_2(birth_data)
    # zad5_2(birth_data)
    # zad6_2(birth_data)
    # zad7_2(birth_data)
    # zad8_2(birth_data)
    # zad9_2(birth_data)
    # zad10_2(birth_data)
    # zad11_2(birth_data)
    death_data = zad12()
    # zad13(birth_data, death_data)
    zad14(birth_data, death_data)
    # zad15(birth_data, death_data)



if __name__ == '__main__':
    main()











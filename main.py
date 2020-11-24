import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import sqlite3
from pandas import DataFrame
import operator
import math


def zad1_2():
    ''' Wczytaj dane ze wszystkich plików do pojedynczej tablicy (używając Pandas). '''
    # os.chdir(r'names')
    list_of_txt_files = glob.glob('names/*.txt')
    list_of_txt_files.sort()
    columns = [ "Name", "Sex", "Number", "Year"]

    df = pd.DataFrame(columns=columns)

    for name_of_file in list_of_txt_files:
        # print(name_of_file)
        df1 = pd.DataFrame(pd.read_csv(name_of_file, sep=',', header=None))
        df1 = df1.rename(columns={0:"Name", 1:"Sex", 2:"Number"})
        # df1["Year"] = str(name_of_file[3:7])
        # print(str(name_of_file[9:13]))
        df1["Year"] = str(name_of_file[9:13])
        # print(type(df))
        df = df.append(df1)
    data2 = df.pivot(index=['Year', 'Name'], columns=['Sex'], values=['Number'])
    # print(data2)
    return data2
    # return df

def zad2_2(data):
    ''' Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie. '''
    data2 = data.groupby(level=1).sum()
    # print(data2)
    print("Zadanie 2")
    print('Ilosc nadanych unikalnych imion w pelnym okresie:',data2.shape[0])
    # 99444

def zad3_2(data):
    ''' Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie rozróżniając imiona męskie i żeńskie. '''
    # print(data)
    data_female = data.iloc[:,0]
    data_female = data_female.dropna()
    data_female = data_female.groupby(level=1).sum()
    # print(data_female)

    data_male = data.iloc[:,1]
    data_male = data_male.dropna()
    data_male = data_male.groupby(level=1).sum()
    # print(data_male)
    print("Zadanie 3")
    print('Ilosc unikalnych kobiecych imion:', data_female.shape[0])
    # 68332
    print('Ilosc unikalnych meskich imion:', data_male.shape[0])
    # 42054

def zad4_2(data):
    ''' Stwórz nowe kolumny frequency_male i frequency_female i określ popularność każdego z imion w danym każdym
    roku dzieląc liczbę razy, kiedy imię zostało nadane przez całkowita liczbę urodzeń dla danej płci.  '''
    # print(data)
    data2 = data.groupby(level=0).sum()
    # print(data2)
    data_zad4 = data.copy()
    data_zad4['Number','frequency_female'] = data.loc[:,('Number', 'F')]/data2.loc[:,('Number', 'F')]
    data_zad4['Number', 'frequency_male'] = data.loc[:,('Number', 'M')]/data2.loc[:,('Number', 'M')]
    print("Zadanie 4")
    print(data_zad4)

def zad5_2(data):
    '''Określ i wyświetl wykres złożony z dwóch podwykresów, gdzie osią x jest skala czasu, a oś y reprezentuje:
    - liczbę urodzin w danym roku (wykres na górze)
    - stosunek liczby narodzin dziewczynek do liczby narodzin chłopców (wykres na dole)
     którym roku zanotowano najmniejszą, a w którym największą różnicę w liczbie urodzeń między chłopcami a dziewczynkami?
    '''

    list_of_years = list(range(1880, 2020))
    quantity_of_birth = {}
    ratio= {}
    data2 = data.groupby(level=0).sum()
    # print(data2)

    for year in list_of_years:
        # print(year, data2.loc[str(year),('Number', 'F')] , data2.loc[str(year),('Number', 'M')])
        quantity_of_birth[year] = (data2.loc[str(year),('Number', 'F')] + data2.loc[str(year),('Number', 'M')])
        ratio[year] = (data2.loc[str(year),('Number', 'F')].astype(float) / data2.loc[str(year),('Number', 'M')].astype(float))

    print("Zadanie 5")
    print("Najwieksza roznica: ", max(ratio, key=ratio.get))
    print("Najmniejsza roznica: ", min(ratio, key=ratio.get))

    # Najwieksza roznica:  1901
    # Najmniejsza roznica:  1880

    fig, ax = plt.subplots(2, 1)
    fig.suptitle("Zadanie 5")
    ax[0].plot(list_of_years, quantity_of_birth.values(), '--r')
    ax[0].set_xlabel('Rok')
    ax[0].set_ylabel('Liczba urodzin w danym roku')

    ax[1].plot(list_of_years, ratio.values(), '-b')
    ax[1].set_xlabel('Rok')
    ax[1].set_ylabel('Stosunek\n liczby narodzin dziewczynek\n do liczby narodzin chłopców')
    # plt.show()

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
    print("Zadanie 6")
    print("Top female names:")
    top_female_df = pd.DataFrame(top_female(data))
    print(top_female_df)
    # ('Mary', 4128052)
    print("Top male names:")
    top_male_df = pd.DataFrame(top_male(data))
    print(top_male_df)
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
    print("Zadanie 7:")
    print('Mary', special_mary_dict)
    print('Marilin', special_marilin_dict)
    print('James', special_james_dict)
    print('Harry', special_harry_dict)

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

    # plt.show()

    # special_mary_dict {1940: 56206, 1980: 11475, 2019: 2209}
    # special_marilin_dict {1940: 0, 1980: 6, 2019: 7}
    # special_james_dict {1940: 62477, 1980: 39327, 2019: 13087}
    # special_harry_dict {1940: 4679, 1980: 860, 2019: 413}

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
    # print('top_female_', top_female_)
    # print('top_male_', top_male_)

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
    print("Zadanie 8")
    print("Rok z najwieksza roznica:", max(difference, key=difference.get))
    # Rok z najwieksza roznica: 1889

    fig, ax = plt.subplots()
    ax.plot(list_of_years, female_ratio, '-r')
    ax.plot(list_of_years, male_ratio, '-b')
    ax.set_title("Zadanie 8")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Procent w danym roku stanowiły imiona należące do rankingu top1000 [%]")
    ax.legend(['female_ratio [%]', 'male_ratio [%]'], loc='upper right')
    # plt.show()
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

    # print(data)
    list_of_years = list(range(1880, 2020))

    last_letters_df = data.reset_index()
    last_letters_df['Name'] = last_letters_df['Name'].str.strip().str[-1]

    last_letters_df = last_letters_df.groupby(by=["Year", "Name"]).sum()

    del last_letters_df['Year']
    del last_letters_df['Name']


    male_last_letters_df2 = pd.DataFrame()
    male_last_letters_df2['2015'] = last_letters_df.loc['2015', ('Number', 'M')]
    male_last_letters_df2['1960'] = last_letters_df.loc['1960', ('Number', 'M')]
    male_last_letters_df2['1910'] = last_letters_df.loc['1910', ('Number', 'M')]


    male_last_letters_df2['2015 normalize'] = male_last_letters_df2['2015'] / last_letters_df.loc['2015'].loc[:, ("Number",'M')].sum()
    male_last_letters_df2['1960 normalize'] = male_last_letters_df2['1960'] / last_letters_df.loc['1960'].loc[:, ("Number",'M')].sum()
    male_last_letters_df2['1910 normalize'] = male_last_letters_df2['1910'] / last_letters_df.loc['1910'].loc[:, ("Number",'M')].sum()

    male_last_letters_df2 = male_last_letters_df2.fillna(0)
    # print(male_last_letters_df2)
    # print(male_last_letters_df2.sum())

    labels = male_last_letters_df2.index.tolist()
    # print(labels)

    fig, ax = plt.subplots()
    x = np.arange(len(labels))

    barWidth = 0.4
    ax.bar(x - barWidth, male_last_letters_df2['1910 normalize'].values *100, barWidth/2, label='1910')
    ax.bar(x , male_last_letters_df2['1960 normalize'].values *100, barWidth/2, label='1960')
    ax.bar(x + barWidth, male_last_letters_df2['2015 normalize'].values *100, barWidth/2, label='2015')

    ax.set_title('Zadanie 9')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('%')
    ax.legend()

    # plt.show()

    male_last_letters_df2['abs (2015 - 1910)'] = abs(male_last_letters_df2['2015 normalize'] - male_last_letters_df2['1910 normalize'])
    # print(male_last_letters_df2)
    column = male_last_letters_df2['abs (2015 - 1910)']
    max_index = column.idxmax()
    print("Zadanie 9")
    print('Litera dla ktorej wystapil najwiekszy wzrost/spadek:',max_index)

    letters_with_biggest_changes = male_last_letters_df2['abs (2015 - 1910)'].nlargest(3).index.tolist()
    # print(letters_with_biggest_changes)
    # print(last_letters_df)

    # data = data[(data.Age == 0) | (data.Age == 1) | (data.Age == 2) | (data.Age == 3) | (data.Age == 4)]
    biggest_changes_df = last_letters_df.copy()
    #  # print(biggest_changes_df)
    biggest_changes_df.reset_index(inplace=True)

    del biggest_changes_df[('Number', 'F')]


    biggest_changes_df = biggest_changes_df[(biggest_changes_df.Name == letters_with_biggest_changes[0]) | (
                biggest_changes_df.Name == letters_with_biggest_changes[1]) | (
                                                        biggest_changes_df.Name == letters_with_biggest_changes[2])]

    biggest_changes_df = biggest_changes_df.pivot(index=['Year'], columns=['Name'], values=['Number'])
    # print(biggest_changes_df)


    labels = biggest_changes_df[('Number', letters_with_biggest_changes[0])].index.tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    ax.plot( biggest_changes_df[('Number', letters_with_biggest_changes[0])].values, '-r')
    ax.plot( biggest_changes_df[('Number', letters_with_biggest_changes[1])].values, '-b')
    ax.plot( biggest_changes_df[('Number', letters_with_biggest_changes[2])].values, '-k')

    ax.set_title("Zadanie 9")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Popularnosc liter")
    plt.xticks(ticks=x, labels=labels)

    ax.legend([letters_with_biggest_changes[0], letters_with_biggest_changes[1], letters_with_biggest_changes[2]], loc='upper right')
    # plt.show()


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

    print('Zadanie 10. Najpopularniejsze imie nadawane chlopcom i dziewczynka:',data2.index[-1])
    # Najpopularniejsze imie nadawane chlopcom i dziewczynka: James

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
    list_of_years = list(range(1880, 2020))
    print(data)

    data = data.dropna()
    print(data)

    data2 = data.groupby(level=0).sum()
    print(data2)

    data_zad11 = data.copy()
    data_zad11['Number','frequency_female'] = data.loc[:,('Number', 'F')]/data2.loc[:,('Number', 'F')]
    data_zad11['Number', 'frequency_male'] = data.loc[:,('Number', 'M')]/data2.loc[:,('Number', 'M')]
    print(data_zad11)
    data_zad11['Number', 'popularity'] = data_zad11.loc[:,('Number', 'frequency_male')]/(data_zad11.loc[:,('Number', 'frequency_male')] + data_zad11.loc[:,('Number', 'frequency_female')])
    print(data_zad11)

    female_male_names_df = data_zad11.drop(data_zad11[(data_zad11.Number.popularity < 0.3) & (data_zad11.Number.popularity > 0.7)].index)
    # print(female_male_names_df)

    # del female_male_names_df['frequency_female']
    # del female_male_names_df['frequency_male']
    # del female_male_names_df['popularity']

    print(female_male_names_df)
    # female_male_names_df = female_male_names_df.reset_index()
    # print(female_male_names_df)

    list_of_years_to_delete = list(range(1921, 2020))
    list_of_years_to_delete = [str(int) for int in list_of_years_to_delete]
    data_1880_1920 = female_male_names_df.drop(list_of_years_to_delete)
    # print(data_1880_1920)

    list_of_years_to_delete = list(range(1880, 2000))
    list_of_years_to_delete = [str(int) for int in list_of_years_to_delete]
    data_2000_2020 = female_male_names_df.drop(list_of_years_to_delete)
    # print(data_2000_2020)

    # data_1880_1920 = data_1880_1920.reset_index()
    # data_2000_2020 = data_2000_2020.reset_index()

    data_1880_1920_sum = data_1880_1920.groupby(level=0).sum()
    data_2000_2020_sum = data_2000_2020.groupby(level=0).sum()

    print(data_1880_1920_sum)
    print(data_2000_2020_sum)

    data_1880_1920 = data_1880_1920.groupby(level="Name").sum()
    data_2000_2020 = data_2000_2020.groupby(level="Name").sum()

    print(data_1880_1920)
    print(data_2000_2020)

    data_1880_1920['Number','frequency_female'] = data_1880_1920.loc[:,('Number', 'F')]/data_1880_1920_sum.loc[:,('Number', 'F')]
    # data_zad11['Number', 'frequency_male'] = data.loc[:,('Number', 'M')]/data2.loc[:,('Number', 'M')]

    print(data_1880_1920)

def zad12():
    '''
    Wczytaj dane z bazy opisującej śmiertelność w okresie od 1959-2018r w poszczególnych grupach wiekowych: USA_ltper_1x1.sqlite,
    opis: https://www.mortality.org/Public/ExplanatoryNotes.php.
    Spróbuj zagregować dane już na etapie zapytania SQL.
    '''
    # os.chdir('..')  # back to main folder
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")  # połączenie do bazy danych - pliku
    c = conn.cursor()

    cursor = conn.execute('SELECT * from USA_fltper_1x1')
    column_names = list(map(lambda x: x[0], cursor.description))

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

    birth_data_sum = birth_data_sum.sum(axis = 1, skipna = True)

    birthrate = {}

    for idx , year in enumerate(years):
        # print(year, birth_data_sum[idx], dff['dx'][idx])
        birthrate[year] = int(birth_data_sum[idx]-dff['dx'][idx])
        pass

    # print(birthrate)
    fig, ax = plt.subplots()
    ax.plot(birthrate.keys(), birthrate.values(), '-r')
    # ax.set_xticklabels([1959, 2017, step=10])
    ax.set_title("Zadanie 13")
    # plt.show()

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
    ax.legend(['przezywalnosc', 'przezywalnosc_5l'], loc='upper right')
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
    birth_data = zad1_2()
    zad2_2(birth_data)
    zad3_2(birth_data)
    zad4_2(birth_data)
    zad5_2(birth_data)
    zad6_2(birth_data)
    zad7_2(birth_data)
    zad8_2(birth_data)
    zad9_2(birth_data)
    zad10_2(birth_data)
    # zad11_2(birth_data)
    death_data = zad12()
    zad13(birth_data, death_data)
    zad14(birth_data, death_data)
    zad15(birth_data, death_data)



if __name__ == '__main__':
    main()











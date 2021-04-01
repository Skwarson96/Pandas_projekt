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
    list_of_txt_files = glob.glob('names/*.txt')
    list_of_txt_files.sort()
    columns = [ "Name", "Sex", "Number", "Year"]

    df = pd.DataFrame(columns=columns)

    for name_of_file in list_of_txt_files:
        df1 = pd.DataFrame(pd.read_csv(name_of_file, sep=',', header=None))
        df1 = df1.rename(columns={0:"Name", 1:"Sex", 2:"Number"})
        df1["Year"] = str(name_of_file[9:13])
        df = df.append(df1)
    data2 = df.pivot(index=['Year', 'Name'], columns=['Sex'], values=['Number'])

    return data2


def zad2_2(data):
    ''' Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie. '''
    data2 = data.groupby(level=1).sum()

    print("Zadanie 2")
    print('Ilosc nadanych unikalnych imion w pelnym okresie:',data2.shape[0])
    # 99444

def zad3_2(data):
    ''' Określi ile różnych (unikalnych) imion zostało nadanych w tym czasie rozróżniając imiona męskie i żeńskie. '''
    data_female = data.iloc[:,0]
    data_female = data_female.dropna()
    data_female = data_female.groupby(level=1).sum()

    data_male = data.iloc[:,1]
    data_male = data_male.dropna()
    data_male = data_male.groupby(level=1).sum()
    print("Zadanie 3")
    print('Ilosc unikalnych kobiecych imion:', data_female.shape[0])
    # 68332
    print('Ilosc unikalnych meskich imion:', data_male.shape[0])
    # 42054

def zad4_2(data):
    ''' Stwórz nowe kolumny frequency_male i frequency_female i określ popularność każdego z imion w danym każdym
    roku dzieląc liczbę razy, kiedy imię zostało nadane przez całkowita liczbę urodzeń dla danej płci.  '''
    data2 = data.groupby(level=0).sum()

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

    for year in list_of_years:
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

    # sort all female values
    data_female = data.sort_values(by=['Year', ('Number', 'F')], ascending=False)

    # find top 1000 for every year
    data_female2 = data_female.groupby(level=0).head(1000)

    # sum number of names for all time
    data_female2 = data_female2.groupby(level="Name").sum()

    # sort values and take first 1000
    top_1000_female = data_female2.sort_values(by=[('Number', 'F')], ascending=False).head(1000)

    top_1000_female = top_1000_female.loc[:, ('Number', 'F')]

    return top_1000_female

def top_male(data):
    # sort all male values
    data_male = data.sort_values(by=['Year', ('Number', 'M')], ascending=False)

    # find top 1000 for every year
    data_male2 = data_male.groupby(level=0).head(1000)

    # sum number of names for all time
    data_male2 = data_male2.groupby(level="Name").sum()

    # sort values and take first 1000
    top_1000_male = data_male2.sort_values(by=[('Number', 'M')], ascending=False).head(1000)

    top_1000_male = top_1000_male.loc[:, ('Number', 'M')]

    return top_1000_male

def zad6_2(data):
    '''Wyznacz 1000 najpopularniejszych imion dla każdej płci w całym zakresie czasowym,
     metoda powinna polegać na wyznaczeniu 1000 najpopularniejszych imion dla każdego roku i dla każdej płci a
     następnie ich zsumowaniu w celu ustalenia rankingu top 1000 dla każdej płci.'''

    print("Zadanie 6")

    print("Top female names:")
    top_female_df = top_female(data)
    print(top_female_df)
    # ('Mary', 4128052)
    print("Top male names:")
    top_male_df = top_male(data)
    print(top_male_df)
    # ('James', 5177716)

    return top_female_df, top_male_df

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

def zad8_2(data, top_female_, top_male_):
    '''
    Wykreśl wykres z podziałem na lata i płeć zawierający informację jaki procent w danym roku
    stanowiły imiona należące do rankingu top1000. Wykres ten opisuje różnorodność imion,
    zanotuj rok w którym zaobserwowano największą różnicę w różnorodności między imionami męskimi a żeńskimi.
    '''
    years = data.index.get_level_values(0).unique()

    # female
    top_female_by_year = data.loc[(years, top_female_.index),('Number', 'F')]
    top_female_sum_by_year = top_female_by_year.groupby(level=0).sum()
    all_female_by_year = data.loc[years, ('Number', 'F')]
    all_female_sum_by_year = all_female_by_year.groupby(level=0).sum()
    female_ratio = top_female_sum_by_year / all_female_sum_by_year

    # male
    top_male_by_year = data.loc[(years, top_male_.index),('Number', 'M')]
    top_male_sum_by_year = top_male_by_year.groupby(level=0).sum()
    all_male_by_year = data.loc[years, ('Number', 'M')]
    all_male_sum_by_year = all_male_by_year.groupby(level=0).sum()
    male_ratio = top_male_sum_by_year / all_male_sum_by_year

    df_difference = np.abs(female_ratio - male_ratio)
    print("Year with biggest difference: ", df_difference.idxmax())


    plt.figure()
    plt.plot(years, female_ratio)
    plt.plot(years, male_ratio)
    plt.legend(['Female', "Male"])
    plt.xlim(left=np.min(years.values), right=np.max(years.values))

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

    labels = male_last_letters_df2.index.tolist()

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


    male_last_letters_df2['abs (2015 - 1910)'] = abs(male_last_letters_df2['2015 normalize'] - male_last_letters_df2['1910 normalize'])

    column = male_last_letters_df2['abs (2015 - 1910)']
    max_index = column.idxmax()
    print("Zadanie 9")
    print('Litera dla ktorej wystapil najwiekszy wzrost/spadek:',max_index)

    letters_with_biggest_changes = male_last_letters_df2['abs (2015 - 1910)'].nlargest(3).index.tolist()

    biggest_changes_df = last_letters_df.copy()
    biggest_changes_df.reset_index(inplace=True)

    del biggest_changes_df[('Number', 'F')]


    biggest_changes_df = biggest_changes_df[(biggest_changes_df.Name == letters_with_biggest_changes[0]) | (
                biggest_changes_df.Name == letters_with_biggest_changes[1]) | (
                                                        biggest_changes_df.Name == letters_with_biggest_changes[2])]

    biggest_changes_df = biggest_changes_df.pivot(index=['Year'], columns=['Name'], values=['Number'])

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

    data2 = data2.sum(axis=1)

    data2 = data2.groupby(level="Name").sum()

    data2 = data2.sort_values()

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

    data = data.dropna()

    data2 = data.groupby(level=0).sum()

    data_zad11 = data.copy()
    data_zad11['Number','frequency_female'] = data.loc[:,('Number', 'F')]/data2.loc[:,('Number', 'F')]
    data_zad11['Number', 'frequency_male'] = data.loc[:,('Number', 'M')]/data2.loc[:,('Number', 'M')]
    data_zad11['Number', 'popularity'] = data_zad11.loc[:,('Number', 'frequency_male')]/(data_zad11.loc[:,('Number', 'frequency_male')] + data_zad11.loc[:,('Number', 'frequency_female')])

    female_male_names_df = data_zad11.drop(data_zad11[(data_zad11.Number.popularity < 0.3) & (data_zad11.Number.popularity > 0.7)].index)

    list_of_years_to_delete = list(range(1921, 2020))
    list_of_years_to_delete = [str(int) for int in list_of_years_to_delete]
    data_1880_1920 = female_male_names_df.drop(list_of_years_to_delete)

    list_of_years_to_delete = list(range(1880, 2000))
    list_of_years_to_delete = [str(int) for int in list_of_years_to_delete]
    data_2000_2020 = female_male_names_df.drop(list_of_years_to_delete)

    data_1880_1920_sum = data_1880_1920.sum()
    data_2000_2020_sum = data_2000_2020.sum()

    data_1880_1920 = data_1880_1920.reset_index()

    data_1880_1920 = data_1880_1920.groupby(by=["Year", "Name"]).sum()
    del data_1880_1920['Year']
    del data_1880_1920['Name']

    data_1880_1920 = data_1880_1920.groupby(level=["Name"]).sum()

    data_1880_1920['Number','frequency_female'] = data_1880_1920.loc[:,('Number', 'F')]/data_1880_1920_sum.loc[('Number', 'F')]
    data_1880_1920['Number', 'frequency_male'] = data_1880_1920.loc[:,('Number', 'M')]/data_1880_1920_sum.loc[('Number', 'M')]
    data_1880_1920['Number', 'popularity'] = data_1880_1920.loc[:, ('Number', 'frequency_male')] / (
                data_1880_1920.loc[:, ('Number', 'frequency_male')] + data_1880_1920.loc[:, ('Number', 'frequency_female')])
    data_1880_1920 = data_1880_1920.reset_index()


    data_2000_2020 = data_2000_2020.reset_index()

    data_2000_2020 = data_2000_2020.groupby(by=["Year", "Name"]).sum()
    del data_2000_2020['Year']
    del data_2000_2020['Name']

    data_2000_2020 = data_2000_2020.groupby(level=["Name"]).sum()

    data_2000_2020['Number', 'frequency_female'] = data_2000_2020.loc[:, ('Number', 'F')] / data_2000_2020_sum.loc[
        ('Number', 'F')]
    data_2000_2020['Number', 'frequency_male'] = data_2000_2020.loc[:, ('Number', 'M')] / data_2000_2020_sum.loc[
        ('Number', 'M')]
    data_2000_2020['Number', 'popularity'] = data_2000_2020.loc[:, ('Number', 'frequency_male')] / (
            data_2000_2020.loc[:, ('Number', 'frequency_male')] + data_2000_2020.loc[:, ('Number', 'frequency_female')])
    data_2000_2020 = data_2000_2020.reset_index()

    names_fom_two_time_frames = data_1880_1920[data_1880_1920.Name.isin(data_2000_2020.Name)]

    data_1880_1920.set_index('Name', inplace=True)
    data_2000_2020.set_index('Name', inplace=True)

    data_names = pd.DataFrame()
    data_names['Name'] = names_fom_two_time_frames['Name']

    list_names = []
    for name in names_fom_two_time_frames['Name']:

         list_names.append(abs(data_1880_1920.loc[name, ('Number', 'popularity')] - data_2000_2020.loc[name, ('Number', 'popularity')]))
    data_names['popularity'] = list_names

    most_popular_female_male_names = data_names['popularity'].nlargest(2).index.tolist()
    print("Zadanie 11")
    print(data_names.loc[most_popular_female_male_names[0], 'Name'], data_names.loc[most_popular_female_male_names[1], 'Name'])

def zad12():
    '''
    Wczytaj dane z bazy opisującej śmiertelność w okresie od 1959-2018r w poszczególnych grupach wiekowych: USA_ltper_1x1.sqlite,
    opis: https://www.mortality.org/Public/ExplanatoryNotes.php.
    Spróbuj zagregować dane już na etapie zapytania SQL.
    '''
    conn = sqlite3.connect("USA_ltper_1x1.sqlite")  # połączenie do bazy danych - pliku

    female_df = pd.read_sql_query('SELECT * FROM USA_fltper_1x1', conn)
    male_df = pd.read_sql_query('SELECT * FROM USA_mltper_1x1', conn)

    data = pd.concat([female_df, male_df], axis=0)

    conn.close()

    return data

def zad13(birth_data, death_data):
    '''
    Wyznacz przyrost naturalny w analizowanym okresie (analizowany okres: 1959-2017)
    '''
    # przyrost naturalny = liczba urodzen - liczba zgonow
    # stopa przyrostu naturalnego = (liczba urodzen - liczba zgonow)/liczba mieszkancow

    # suma po latach ( lata jako indeksy)
    birth_data_sum = birth_data.groupby(["Year"]).sum()
    # suma kobiet i mezczyzn
    birth_data_sum = birth_data_sum.sum(axis = 1, skipna = True)

    # suma po latach ( lata jako indeksy)
    dff = death_data.groupby(["Year"]).dx.sum()

    # ograniczenie danych do analizowanego okresu
    birth_data_sum2 = birth_data_sum.loc[(birth_data_sum.index >= '1959') & (birth_data_sum.index <= '2017')]
    dff2 = dff.loc[(dff.index >= 1959) & (dff.index <= 2017)]

    # obliczenie przyrostu
    birthrate = birth_data_sum2.values - dff2.values

    # wykres:
    fig, ax = plt.subplots()
    ax.plot(birth_data_sum2.index, birthrate, '-r')

    ax.set_title("Zadanie 13")
    # plt.show()

def zad14(birth_data, death_data):
    '''
    Wyznacz i wyświetl współczynnik przeżywalności dzieci w pierwszym roku życia
    '''
    dff = death_data.groupby(["Year"]).dx.sum().reset_index()

    years = list(dff['Year'])
    years = [str(int) for int in years]

    birth_data_sum = birth_data.groupby(["Year"]).sum().reset_index()

    birth_data_sum = birth_data_sum[birth_data_sum["Year"].isin(years)]

    birth_data_sum = birth_data_sum.reset_index()


    birth_data_sum.pop("index")
    birth_data_sum['Sum'] = birth_data_sum[('Number','F')] + birth_data_sum[('Number','M')]

    death_data_for_0 = death_data[death_data.Age == 0].reset_index()
    death_data_for_0.pop("index")

    death_data_sum = death_data_for_0.groupby(['Year']).sum().reset_index()

    birth_data_sum['D'] = death_data_sum['dx']

    birth_data_sum['przezywalnosc'] = (birth_data_sum['Sum'] - birth_data_sum['D']) / birth_data_sum['Sum'] * 100

    zad15_data = zad15(birth_data_sum, death_data)

    zad15_data[2014] = None
    zad15_data[2015] = None
    zad15_data[2016] = None
    zad15_data[2017] = None

    birth_data_sum['D5'] = zad15_data.values()

    birth_data_sum['przezywalnosc_5l'] = (birth_data_sum['Sum'] - birth_data_sum['D5']) / birth_data_sum['Sum'] * 100


    fig, ax = plt.subplots()
    ax.plot(birth_data_sum['przezywalnosc'], '-r')
    ax.plot(birth_data_sum['przezywalnosc_5l'], '-b')

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
    years = list(dff['Year'])

    data_dict = {}

    data = death_data[['Sex', 'Year', 'Age', 'dx']]

    data = data[(data.Age == 0) | (data.Age == 1) | (data.Age == 2) | (data.Age == 3) | (data.Age == 4)]

    data = data.groupby(by=["Year", "Age"]).sum()


    for year in years:
        try:
            data_dict[year] = int(data.loc[year, 0].values[0]) + \
                              int(data.loc[year + 1, 1].values[0]) + \
                              int(data.loc[year + 2, 2].values[0]) + \
                              int(data.loc[year + 3, 3].values[0]) + \
                              int(data.loc[year + 4, 4].values[0])
        except KeyError:
            pass


    return data_dict

def main():
    birth_data = zad1_2()
    zad2_2(birth_data)
    zad3_2(birth_data)
    zad4_2(birth_data)
    zad5_2(birth_data)
    f_1000, m_1000 = zad6_2(birth_data)
    zad7_2(birth_data)
    zad8_2(birth_data, f_1000, m_1000)
    zad9_2(birth_data)
    zad10_2(birth_data)
    zad11_2(birth_data)
    death_data = zad12()
    zad13(birth_data, death_data)
    zad14(birth_data, death_data)
    zad15(birth_data, death_data)



if __name__ == '__main__':
    main()

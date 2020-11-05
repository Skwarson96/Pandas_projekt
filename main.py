import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from pandas import DataFrame



def zad1():
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
    
def main():
    data = zad1()
    # zad2(data)
    zad3(data)



if __name__ == '__main__':
    main()











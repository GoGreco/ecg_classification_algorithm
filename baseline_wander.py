import os
import pandas as pd
import matplotlib.pyplot as plt


def load_record_names(record_file_path):

    with open(record_file_path, 'r') as file:
        record_name = file.read().splitlines()
    return record_name

def load_csv_to_df(file_name, file_path = './signal_tables/'):
    dataFrame = pd.read_csv(f'{file_path}/{file_name}.csv')
    return dataFrame



if __name__ == '__main__':
    #importing all the signals
    names_path = './database/RECORDS'
    tables_path = './signal_tables/'

    record_dataFrame = []
    annotation_dataFrame =[]

    record_number= load_record_names(names_path)

    for i in range(len(record_number)):
        record_name = f'{record_number[i]}_record'
        annotation_name =f'{record_number[i]}_annotation'

        record_dataFrame.append(load_csv_to_df(record_name, tables_path))
        annotation_dataFrame.append(load_csv_to_df(annotation_name, tables_path))

    record_dataFrame[0].plot()
    plt.title("Record 100")
    plt.savefig('./graphs')

#pip install -r requirements_data_load.txt
import os
import wfdb   
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt 

#Using the same method of reading the RECORDS file for the names:
def load_records_names(record_file_path):

    with open(record_file_path, 'r') as file:
        record_name = file.read().splitlines()

    return record_name


#carrega cada sinal como um pandas dataframe
def load_as_df(record_number, record_path = "./signal_tables"):
    record_path = f'{record_path}/{record_number}_record.csv'
    data_frame = pd.read_csv(record_path)

    return data_frame


if __name__ == '__main__':

    #variaveis
    sampling_rate = 360

    #variaveis de local 
    database_path  = './database'
    signal_tables  = './signal_tables'
    processed_path = './filtered_tables'
    
    name_paths = f"{database_path}/RECORDS"

    record_names  = load_records_names(name_paths)

    records = []


    for name in record_names:
        records.append(load_as_df(name, signal_tables))
    

    filtered_dataframe = pd.DataFrame({"Raw": records[0],
        "bessel": nk.signal_filter(records[0], highcut=3, method="bessel", order=5),
    }).plot()

    plt.show()
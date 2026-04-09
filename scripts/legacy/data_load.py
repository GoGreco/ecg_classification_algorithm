#pip install -r requirements_data_load.txt
import os
import wfdb
import pandas as pd

#Using the same method of reading the RECORDS file for the names:
def load_records_names(record_file_path):

    with open(record_file_path, 'r') as file:
        record_name = file.read().splitlines()

    return record_name

#creating the dataframes:
def create_record_df(signal_name, file_path ='./database'):
    signal, channels = wfdb.rdsamp(f'{file_path}/{signal_name}')

    signal_frame = pd.DataFrame(signal, columns = channels['sig_name'])
    return signal_frame

def create_annotation_df(signal_name, file_path ='/database'):
    annotations = wfdb.rdann(f'{file_path}/{signal_name}', 'atr')

    df_model ={
        'Sample ': annotations.sample,
        'Symbol ': annotations.symbol
    }
    
    ann_df = pd.DataFrame(df_model)

    return ann_df


#saving the dataframes to csv
def save_record(dataframe, signal_name, saving_path = './signal_tables'):

    place_and_name = f'{saving_path}/{signal_name}_record.csv'
    if os.path.exists(place_and_name):
        print('File already exists')
        return
    else:
        dataframe.to_csv(place_and_name, index=False)
        print(f'Saved the record: {signal_name} in {saving_path}')

def save_ann(dataframe, signal_name, saving_path = './signal_tables'):

    place_and_name = f'{saving_path}/{signal_name}_annotation.csv'
    if os.path.exists(place_and_name):
        print('File already exists')
        return
    else:
        dataframe.to_csv(place_and_name, index=False)
        print(f'Saved the record: {signal_name} in {saving_path}')


if __name__ == '__main__':
    
    data_path = './database'
    processed_path = './signal_tables'
    record_names = []


    record_names = load_records_names(f'{data_path}/RECORDS')
    

    for name in record_names:
        record_frame = create_record_df(name, data_path)
        save_record(record_frame, name, processed_path)

        annotation_frame = create_annotation_df(name, data_path)
        save_ann(annotation_frame, name, processed_path)


    


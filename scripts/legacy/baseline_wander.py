import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

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

    record_number= load_record_names(names_path)
    
    record_dataFrames ={}

    #Loading all records


    record_name = f'{record_number[0]}_record'

    dataFrame = load_csv_to_df(record_name, tables_path)

    if len(dataFrame.columns) >= 2:
        dataFrame.columns = ["MLII", "V1"]

    record_dataFrames[record_number[0]] = dataFrame


    #testing the baseline wander
    full_signal = record_dataFrames['100']['MLII']

    fq =360
    n = 40000
    t = np.arange(n)/fq
    dirty_signal = full_signal[:n]

    dirtier_signal = dirty_signal +0.5*np.sin(2*np.pi*0.3*t)+ 0.08 * np.random.randn(n)

    #to show the contaminated signal
    time = np.arange(dirtier_signal.size)/fq
    plt.figure(figsize=(12,6))
    plt.plot(time, dirty_signal, label='Original Signal', color='green', linewidth=2)
    plt.plot(time, dirtier_signal, label='Contaminated Signal', color='red', alpha = 0.5)
    plt.title("Original Vs Contaminated ECG Signals")
    plt.xlabel("Tempo em Segundos")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


    ripple = "sym10"
    fc = 0.5
    target = int(np.ceil(np.log2(fq/fc)))
    max_level = pywt.dwt_max_level(n, pywt.Wavelet(ripple).dec_len)
    lev = min(target, max_level)

    coefficient = pywt.wavedec(dirtier_signal, ripple, level=lev)

    coefficient_approx = [coefficient[0]]+ [np.zeros_like(c) for c in coefficient[1:]]
    baseline = pywt.waverec(coefficient_approx, ripple)[:n]

    signal_wavelet_baseline = dirtier_signal- baseline

    plt.figure(figsize=(12,6))
    plt.plot(time, dirtier_signal, label='Original Signal', color='red', alpha = 0.5)
    plt.plot(time, baseline, label='Contaminated Bbaseline', color='blue', linewidth=2)
    plt.title("Baseline from the Contaminated Signal")
    plt.xlabel("Tempo em Segundos")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(time, signal_wavelet_baseline, label='Original Signal', color='orange')
    plt.title("Baseline from the Contaminated Signal")
    plt.xlabel("Tempo em Segundos")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

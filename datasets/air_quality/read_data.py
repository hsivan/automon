import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_all_dataset_columns():
    df = pd.read_csv('PRSA_Data_Aotizhongxin_20130301-20170228.csv')
    print(df.columns)
    df = df.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd', 'station'])
    df = df.dropna()
    print(df.columns)
    x = np.arange(0, df['TEMP'].shape[0])
    print(x)

    fig, axs = plt.subplots(11, 1)

    for idx, column in enumerate(df.columns):
        axs[idx].plot(x, df[column], '-', label=column)
        axs[idx].legend()

    plt.savefig('correlation_between_columns.pdf')
    plt.show()


def read_csv(csv_file):
    df_station = pd.read_csv(csv_file)
    df_station = df_station.drop(columns=['No', 'year', 'month', 'day', 'hour', 'wd', 'station'])
    df_station = df_station.dropna()
    return df_station


def prepare_pm_data(step=20, relative_folder='./'):
    csv_files = [csv_file for csv_file in os.listdir(relative_folder) if csv_file.endswith('csv')]
    station_data_arr = []

    for csv_file in csv_files:
        station_name = csv_file.split('_')[2]
        print("station_name:", station_name)
        df_station = read_csv(relative_folder + csv_file)
        # x vector is the TEMP (0 to 25) ans y vector is the DEWP (dew point temperature, -25 to 25)

        number_of_data_point = df_station['PM10'].shape[0]
        data = np.zeros((number_of_data_point, 2), dtype='int')
        x = df_station['PM10'].values
        y = df_station['PM2.5'].values

        x = x.clip(0, 500)
        bins_temp = np.arange(0, 500, step)
        indices_temp = np.digitize(x, bins_temp) - 1
        x = indices_temp

        y = y.clip(0, 500)
        bins_dewp = np.arange(0, 500, step)
        indices_dewp = np.digitize(y, bins_dewp) - 1
        y = indices_dewp

        data[:, 0] = x
        data[:, 1] = y
        station_data_arr.append(data)

    return station_data_arr


if __name__ == "__main__":
    plot_all_dataset_columns()
    prepare_pm_data()

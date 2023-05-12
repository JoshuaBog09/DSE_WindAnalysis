import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime

def main():
    head = ["DTG",
            "F010", "D010", "T010", "Q010", "P010",
            "F020", "D020", "T020", "Q020", "P020",
            "F040", "D040", "T040", "Q040", "P040",
            "F060", "D060", "T060", "Q060", "P060",
            "F080", "D080", "T080", "Q080", "P080",
            "F100", "D100", "T100", "Q100", "P100",
            "F150", "D150", "T150", "Q150", "P150",
            "F200", "D200", "T200", "Q200", "P200"]

    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M')

    f = pd.read_csv('Data/MMIJ_2004-2013.csv', delimiter='\t', comment='#', encoding='utf-8', header=None, names=head,
                    index_col=False, parse_dates=['DTG'], date_parser=dateparse)
    df = f.set_index('DTG')

    '''Specify seasons'''
    df_spring = df[((df.index.month == 3) & (df.index.day >= 20)) | (
                (df.index.month == 4) | (df.index.month == 5) | ((df.index.month == 6) & (df.index.day < 21)))]

    df_summer = df[((df.index.month == 6) & (df.index.day >= 21)) | (
            (df.index.month == 7) | (df.index.month == 8) | ((df.index.month == 9) & (df.index.day < 23)))]

    df_fall = df[((df.index.month == 9) & (df.index.day >= 23)) | (
            (df.index.month == 10) | (df.index.month == 11) | ((df.index.month == 12) & (df.index.day < 22)))]

    df_winter = df[((df.index.month == 12) & (df.index.day >= 22)) | (
            (df.index.month == 1) | (df.index.month == 2) | ((df.index.month == 3) & (df.index.day < 20)))]


    '''Define function to plot'''
    def plot_hist_weibull_alt10(data):
        alt = data['F010'].sort_values()
        para1, para2, para3 = stats.weibull_min.fit(alt, floc=0)
        alt.hist(alpha = 0.4, density = True, bins = 30)
        plt.plot(alt, stats.weibull_min.pdf(alt, para1, para2, para3), color ='blue', label = '10m Altitude')
        plt.axvline(np.median(alt), color='blue', linestyle='dashed', linewidth=1)

    def plot_hist_weibull_alt200(data):
        alt = data['F200'].sort_values()
        para1, para2, para3 = stats.weibull_min.fit(alt, floc=0)
        alt.hist(alpha=0.4, density=True, bins=30)
        plt.plot(alt, stats.weibull_min.pdf(alt, para1, para2, para3), color = 'red', label = '200m ALtitude')
        plt.axvline(np.median(alt), color='red', linestyle='dashed', linewidth=1)

    def plot_hist_weibull_season(data):
        plot_hist_weibull_alt10(data)
        plot_hist_weibull_alt200(data)
        plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(25, color='k', linestyle='dashed', linewidth=1)
        plt.legend()
        plt.show()

    def plot_hist_weibull_alt010_2(data1, data2, data3, data4):
        alt_010_1 = data1['F010'].sort_values()
        para11, para21, para31 = stats.weibull_min.fit(alt_010_1, floc=0)
        alt_010_2 = data2['F010'].sort_values()
        para12, para22, para32 = stats.weibull_min.fit(alt_010_2, floc=0)
        alt_010_3 = data3['F010'].sort_values()
        para13, para23, para33 = stats.weibull_min.fit(alt_010_3, floc=0)
        alt_010_4 = data4['F010'].sort_values()
        para14, para24, para34 = stats.weibull_min.fit(alt_010_4, floc=0)
        plt.plot(alt_010_1, stats.weibull_min.pdf(alt_010_1, para11, para21, para31))
        plt.plot(alt_010_2, stats.weibull_min.pdf(alt_010_2, para12, para22, para32))
        plt.plot(alt_010_3, stats.weibull_min.pdf(alt_010_3, para13, para23, para33))
        plt.plot(alt_010_4, stats.weibull_min.pdf(alt_010_4, para14, para24, para34))
        plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(25, color='k', linestyle='dashed', linewidth=1)
        plt.show()

    def plot_hist_weibull_alt200_2(data1, data2, data3, data4):
        alt_010_1 = data1['F200'].sort_values()
        para11, para21, para31 = stats.weibull_min.fit(alt_010_1, floc=0)
        alt_010_2 = data2['F200'].sort_values()
        para12, para22, para32 = stats.weibull_min.fit(alt_010_2, floc=0)
        alt_010_3 = data3['F200'].sort_values()
        para13, para23, para33 = stats.weibull_min.fit(alt_010_3, floc=0)
        alt_010_4 = data4['F200'].sort_values()
        para14, para24, para34 = stats.weibull_min.fit(alt_010_4, floc=0)
        plt.plot(alt_010_1, stats.weibull_min.pdf(alt_010_1, para11, para21, para31))
        plt.plot(alt_010_2, stats.weibull_min.pdf(alt_010_2, para12, para22, para32))
        plt.plot(alt_010_3, stats.weibull_min.pdf(alt_010_3, para13, para23, para33))
        plt.plot(alt_010_4, stats.weibull_min.pdf(alt_010_4, para14, para24, para34))
        plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(25, color='k', linestyle='dashed', linewidth=1)
        plt.show()

    plot_hist_weibull_season(df_spring)
    plot_hist_weibull_season(df_summer)
    plot_hist_weibull_season(df_fall)
    plot_hist_weibull_season(df_winter)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

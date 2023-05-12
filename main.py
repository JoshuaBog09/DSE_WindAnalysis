import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import datetime


class Wind:

    cut_in = 3
    cut_out = 25

    def __init__(self, data, label: str):
        self.data = data
        self.label = label

    def plot_histogram(self, height: str) -> None:

        plt.hist(self.data[f"F{height}"], bins=list(range(0, 35)), density=True, label=f"Alt = {height}", alpha=0.2)
        plt.axvline(Wind.cut_in, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(Wind.cut_out, color='k', linestyle='dashed', linewidth=1)
        plt.xlabel(f"Wind velocity bins")
        plt.ylabel(f"Frequency")

    def plot_velocities(self, height: str) -> None:
        plt.plot(self.data["DTG"], self.data[f"F{height}"])
        # plt.show()

    def plot_velocityprofile(self) -> None:
        altitude = [10, 20, 40, 60, 80, 100, 150, 200]
        velocity = [np.percentile(self.data['F010'], 50), np.percentile(self.data['F020'],50), np.percentile(self.data['F040'],50),
                    np.percentile(self.data['F060'], 50), np.percentile(self.data['F080'],50), np.percentile(self.data['F100'],50),
                    np.percentile(self.data['F150'], 50), np.percentile(self.data['F200'],50)]
        plt.plot(velocity, altitude, label=self.label)

    def plot_velocityprofileV2(self) -> None:

        altitude = [10, 20, 40, 60, 80, 100, 150, 200]
        velocity = [self.data['F010'], self.data['F020'], self.data['F040'],
                    self.data['F060'], self.data['F080'], self.data['F100'],
                    self.data['F150'], self.data['F200']]

        for veloc, alt in zip(velocity, altitude):
            percentage = len(veloc[(3 <= veloc) & (veloc <= 25)]) / len(veloc)
            print(f"for altitude {alt} the percentage is {percentage}")

    def plot_boxplot(self, height: str) -> None:
        # self.data.boxplot(column=[f"F{height}"])
        plt.boxplot(self.data[f"F{height}"])

    @staticmethod
    def show_plots():
        plt.legend()
        plt.show()

    @classmethod
    def cls_from_period(cls, data, start, end, label):
        return cls(data.loc[(data['DTG'] >= start) & (data['DTG'] < end)], label)

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
    # print(f.head())

    # plt.hist(f['F010'], bins=list(range(0, 45)), density=True)
    # plt.show()
    #
    # alt = [10,20,40,60,80,100,150,200]
    # veloc = [np.average(f['F010']), np.average(f['F020']), np.average(f['F040']),
    #          np.average(f['F060']), np.average(f['F080']), np.average(f['F100']),
    #          np.average(f['F150']), np.average(f['F200'])]
    #
    # plt.plot(veloc, alt)
    # plt.show()


    '''Specify Season'''
    # Spring
    start_spring = pd.to_datetime('2004-03-20', format='%Y-%m-%d %H:%M:%S.%f')
    end_spring = pd.to_datetime('2004-06-20', format='%Y-%m-%d %H:%M:%S.%f')

    # Summer
    start_summer = pd.to_datetime('2004-06-21', format='%Y-%m-%d %H:%M:%S.%f')
    end_summer = pd.to_datetime('2004-09-23', format='%Y-%m-%d %H:%M:%S.%f')

    # Fall
    start_fall = pd.to_datetime('2004-09-23', format='%Y-%m-%d %H:%M:%S.%f')
    end_fall = pd.to_datetime('2004-12-22', format='%Y-%m-%d %H:%M:%S.%f')

    # Winter
    start_winter = pd.to_datetime('2004-12-22', format='%Y-%m-%d %H:%M:%S.%f')
    end_winter = pd.to_datetime('2005-03-20', format='%Y-%m-%d %H:%M:%S.%f')

    spring = Wind.cls_from_period(f, start_spring, end_spring, "spring")
    summer = Wind.cls_from_period(f, start_summer, end_summer, "summer")
    fall = Wind.cls_from_period(f, start_fall, end_fall, "fall")
    winter = Wind.cls_from_period(f, start_winter, end_winter, "winter")

    spring.plot_histogram("010")
    spring.plot_histogram("200")
    spring.show_plots()

    summer.plot_histogram("010")
    summer.plot_histogram("200")
    summer.show_plots()

    fall.plot_histogram("010")
    fall.plot_histogram("200")
    fall.show_plots()

    winter.plot_histogram("010")
    winter.plot_histogram("200")
    winter.show_plots()

    # spring.plot_velocities("010")

    spring.plot_velocityprofile()
    summer.plot_velocityprofile()
    fall.plot_velocityprofile()
    winter.plot_velocityprofile()

    spring.plot_velocityprofileV2()
    summer.plot_velocityprofileV2()
    fall.plot_velocityprofileV2()
    winter.plot_velocityprofileV2()

    plt.legend()
    plt.show()

    new = pd.DataFrame([spring.data["F010"], summer.data["F010"], fall.data["F010"], winter.data["F010"]]).transpose()
    new.set_axis(['Spring', 'Summer', 'Fall', 'Winter'], axis=1, inplace=True)

    new.boxplot(column=["Spring", "Summer", "Fall", "Winter"])
    plt.ylabel("Wind velocity [m/s]")
    plt.show()

    new = pd.DataFrame([spring.data["F200"], summer.data["F200"], fall.data["F200"], winter.data["F200"]]).transpose()
    new.set_axis(['Spring', 'Summer', 'Fall', 'Winter'], axis=1, inplace=True)

    new.boxplot(column=["Spring", "Summer", "Fall", "Winter"])
    plt.ylabel("Wind velocity [m/s]")
    plt.show()

    df_spring = df[((df.index.month == 3) & (df.index.day >= 20)) | (
                (df.index.month == 4) | (df.index.month == 5) | ((df.index.month == 6) & (df.index.day < 21)))]

    df_summer = df[((df.index.month == 6) & (df.index.day >= 21)) | (
            (df.index.month == 7) | (df.index.month == 8) | ((df.index.month == 9) & (df.index.day < 23)))]

    df_fall = df[((df.index.month == 9) & (df.index.day >= 23)) | (
            (df.index.month == 10) | (df.index.month == 11) | ((df.index.month == 12) & (df.index.day < 22)))]

    df_winter = df[((df.index.month == 12) & (df.index.day >= 22)) | (
            (df.index.month == 1) | (df.index.month == 2) | ((df.index.month == 3) & (df.index.day < 20)))]


    # Analysis of windspeed over time (year)
    start_year = np.where((df.index.month == 1) & (df.index.day == 1) & (df.index.hour == 00) & (df.index.minute == 00))[0]

    h_axis = []
    average_f010 = []
    average_f200 = []

    for year_idx in range(0, len(start_year) - 1):
        average_f010.append(np.average((df.iloc[start_year[year_idx]:start_year[year_idx + 1]])['F010']))
        average_f200.append(np.average((df.iloc[start_year[year_idx]:start_year[year_idx + 1]])['F200']))
        h_axis.append(df.index[start_year[year_idx]])

    plt.plot(h_axis, average_f010)
    plt.plot(h_axis, average_f200)
    plt.show()

    # Analysis of windspeed over time (month)
    start_month = np.where((df.index.day == 1) & (df.index.hour == 00) & (df.index.minute == 00))[0]

    h_axis = []
    average_f010 = []
    average_f200 = []

    for month_idx in range(0, len(start_month) - 1):
        average_f010.append(np.average((df.iloc[start_month[month_idx]:start_month[month_idx + 1]])['F010']))
        average_f200.append(np.average((df.iloc[start_month[month_idx]:start_month[month_idx + 1]])['F200']))
        h_axis.append(df.index[start_month[month_idx]])

    plt.plot(h_axis, average_f010)
    plt.plot(h_axis, average_f200)
    plt.show()



    # df.iloc[start_year]

    # # Group the DataFrame by year and store in a dictionary
    # df_spring_year = {year: group for year, group in df_spring.groupby(pd.Grouper(freq='Y'))}
    # print(df_summer,'\n',df_fall,'\n', df_winter)

    '''Same season different altitude in one plot'''
    df_spring['F010'].hist(alpha = 0.4, density = True)
    df_spring['F200'].hist(alpha = 0.4, density = True)
    plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(25, color='k', linestyle='dashed', linewidth=1) #Joshua Bogaert's
    plt.show()

    df_summer['F010'].hist(alpha = 0.4, density = True)
    df_summer['F200'].hist(alpha = 0.4, density = True)
    plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(25, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    df_fall['F010'].hist(alpha = 0.4, density = True)
    df_fall['F200'].hist(alpha = 0.4, density = True)
    plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(25, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    df_winter['F010'].hist(alpha = 0.4, density = True)
    df_winter['F200'].hist(alpha = 0.4, density = True)
    plt.axvline(3, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(25, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    '''Same altitude different seasons in one plot'''
    # xmin = 0
    # xmax = 35
    # x = np.linspace(xmin, xmax, 1000)
    # y_spring_10 = scipy.stats.gaussian_kde(df_spring['F010'])
    # y_summer_10 = scipy.stats.gaussian_kde(df_summer['F010'])
    # y_fall_10 = scipy.stats.gaussian_kde(df_fall['F010'])
    # y_winter_10 = scipy.stats.gaussian_kde(df_winter['F010'])
    #
    # plt.plot(y_spring_10)
    # plt.plot(y_summer_10)
    # plt.plot(y_fall_10)
    # plt.plot(y_winter_10)
    # plt.show()

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

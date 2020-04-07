""" Wed Apr 24 09:09:08 2019; By Rojan Shrestha """

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, fo_train_ratio, fo_valid_ratio, fo_test_ratio):
        """ Divide a data into training, validation, and testing set in the given ration
        
        Args: 
            fo_train_ratio : ratio for training dataset
            fo_valid_ratio : ratio for validation dataset
            fo_test_ratio  : ratio for testing dataset 
        """

        self._fo_train_ratio = fo_train_ratio
        self._fo_valid_ratio = fo_valid_ratio
        self._fo_test_ratio = fo_test_ratio

    def convert_time_to_minute(self, st_row):
        """Convert given time into minute
        
        Args:
            st_row: input a time in hour:minute format 
        
        Returns:
            int: return minutes 
        """
        ar_hr_mi_se = st_row.split(":")
        return int(ar_hr_mi_se[0]) * 60 + int(ar_hr_mi_se[1])

    def convert_time_to_hours(self, st_row):
        """ Convert time into hour
        
        Args:
            st_row: input a time in a format hour:minute 
        
        Returns:
            int: return hours ignores decimal 
        """

        ar_hr_mi_se = st_row.split(":")
        return int(ar_hr_mi_se[0]) + int(ar_hr_mi_se[1]) / 60.0

    def get_xy(self, st_row, fo_max_value=360.0):
        """ Convert time into polar coordinate system. This converts 12pm at midnight 
            and 1am in the morning are close to each other. 
        
        Args:
            st_row: input a time in a format hour:minute 
            fo_max_value: Defaults to 360.0. 
        
        Returns:
            pandas series: pair of values - x and y 
        """

        ar_hr_mi_se = st_row.split(":")
        fo_hrs = (float(ar_hr_mi_se[0]) * 60.0 + float(ar_hr_mi_se[1])) / fo_max_value
        fo_r = fo_max_value / (2.0 * np.pi)
        fo_x = fo_r * np.sin(2.0 * np.pi * fo_hrs)
        fo_y = fo_r * np.cos(2.0 * np.pi * fo_hrs)
        return pd.Series([fo_x, fo_y])

    def get_month_xy(self, st_row, fo_max_value=12.0):
        """ This coversion makes january and december are consecutive.
            Therefore, forecasting for these months are very similar.
        
        Args:
            st_row: input a time in hour:minute format 
            fo_max_value: max value, Defaults to 12.0.

        Returns:
            pandas series: pair of values - x and y 
        """

        fo_month = float(st_row) / fo_max_value
        fo_r = fo_max_value / (2.0 * np.pi)
        fo_x = fo_r * np.sin(2.0 * np.pi * fo_month)
        fo_y = fo_r * np.cos(2.0 * np.pi * fo_month)
        return pd.Series([fo_x, fo_y])

    def get_day_xy(self, st_row, fo_max_value=365.0):
        """ Convert 365 days/year to x, y coordinates and 
            31 days/month is rough approximation 
        
        Args:
            st_row: date in format dd-mo-yy 
            fo_max_value: upper bound. Defaults to 365.0.
        
        Raises:
            ValueError: [description]
            ValueError: [description]
        
        Returns:
            pandas series: two elements in a list
        """

        # format dd-mo-yy
        ar_dd_mo_yy = st_row.split("-")
        fo_day = (
            float(ar_dd_mo_yy[0]) + 31.0 * (float(ar_dd_mo_yy[1]) - 1.0)
        ) / fo_max_value

        fo_r = fo_max_value / (2.0 * np.pi)
        fo_x = fo_r * np.sin(2.0 * np.pi * fo_day)
        fo_y = fo_r * np.cos(2.0 * np.pi * fo_day)
        return pd.Series([fo_x, fo_y])

    def encode_month(self, st_row):
        """
    Encode the month into four different categories based on season
    
    Arguments:
        st_row -- month index between 1 to 12 and it raises an error 
                  for outside index
    
    Returns:
        int -- either 1, 2, 3, or 4 
    """

        if st_row < 1 or st_row > 12:
            raise ValueError("Invalid argument")

        if st_row == 1 or st_row == 2 or st_row == 12:
            return 1
        if st_row == 3 or st_row == 11:
            return 2
        if st_row == 4 or st_row == 10:
            return 3
        else:
            return 4

    def encode_hours(self, st_row):
        """
    Convert 24 hours into six groups 
    
    Arguments:
        st_row -- index for each hours between 1 to 24 
    
    Returns:
        int -- index between 1 to 6 
    """

        st_row = int(self.convert_time_to_hours(st_row))
        if st_row < 1 or st_row > 24:
            raise ValueError("Invalid argument")

        if st_row in range(4):
            return 1
        elif st_row in range(4, 8):
            return 2
        elif st_row in range(8, 12):
            return 3
        elif st_row in range(12, 17):
            return 4
        elif st_row in range(17, 20):
            return 5
        elif st_row in range(20, 25):
            return 6

    def parse_excluded_columns(self, st_column_idxes):
        """ 
        Aims: 
            Parsing the column indexes that should be excluded.
        Params:
            st_column_idxes: formats can be 1,2,3 or 1,2-5, or 2-5
    """
        ts_col_idxes = []
        ts_comma = st_column_idxes.split(",")
        for st_comma in ts_comma:
            ts_dash = st_comma.split("-")
            if len(ts_dash) == 1:
                ts_col_idxes.append(int(ts_dash[0]))
            elif len(ts_dash) == 2:
                ts_col_idxes += np.arange(int(ts_dash[0]), int(ts_dash[1]), 1)
        return ts_col_idxes

    def read_data(
        self,
        st_path,
        use_embedded_date=False,
        use_date=False,
        exclude_column_idxes=None,
        exclude_columns=None,
    ):
        """
    Aims: 
      * read CSV file
      * encode year, month, day, and time
      * divide columns into numeric and string based on their content 

    Parameters:
      * st_path: path to input file
      * convert: convert month, day, and time into floating point numeric values 
    """
        print("INFO: reading {}...".format(st_path))
        self._df_data = pd.read_csv(st_path)
        self._df_data["Date Time"] = pd.to_datetime(
            self._df_data["Date Time"], dayfirst=True
        ).astype("datetime64[ns]")

        ts_str_col_names = []
        if use_date:
            # Extracting year, month, day and hour
            self._df_data["Year"] = self._df_data["Date Time"].dt.year
            self._df_data["Month"] = self._df_data["Date Time"].dt.month
            self._df_data["Day"] = self._df_data["Date Time"].dt.day
            self._df_data["Time"] = (
                self._df_data["Date Time"].dt.hour
                + self._df_data["Date Time"].dt.minute / 60.00
                + self._df_data["Date Time"].dt.second / 3600.00
            )

            # # Convert time, month, and day into numeric value
            self._df_data[["Tx", "Ty"]] = (
                self._df_data["Date Time"]
                .dt.time.astype(str)
                .apply(lambda row: self.get_xy(row, 24.0))
            )
            self._df_data[["Mx", "My"]] = (
                self._df_data["Date Time"]
                .dt.month.astype(str)
                .apply(lambda row: self.get_month_xy(row, 12.0))
            )
            self._df_data[["Dx", "Dy"]] = (
                self._df_data["Date Time"]
                .dt.date.astype(str)
                .apply(lambda row: self.get_day_xy(row, 372.0))
            )
            print("INFO: encoded!!!")

        if use_embedded_date:  # donot modify but just split
            # Extracting year, month, day and hour
            self._df_data["Year"] = self._df_data["Date Time"].dt.year
            self._df_data["Month"] = self._df_data["Date Time"].dt.month
            self._df_data["Day"] = self._df_data["Date Time"].dt.day
            self._df_data["Time"] = (
                self._df_data["Date Time"].dt.hour
                + self._df_data["Date Time"].dt.minute / 60.00
                + self._df_data["Date Time"].dt.second / 3600.00
            )
            self._df_data["Year"] = self._df_data["Year"].astype(str)
            self._df_data["Month"] = self._df_data["Month"].astype(str)
            self._df_data["Day"] = self._df_data["Day"].astype(str)
            self._df_data["Time"] = self._df_data["Time"].astype(str)
            ts_str_col_names = ["Year", "Month", "Day", "Time"]

        print("INFO: finished reading!!!")
        print(self._df_data.info())
        print(self._df_data.head())
        print(self._df_data.tail())

        if exclude_column_idxes:
            self._df_data_copy = self._df_data.copy()
            ts_excluded_cols_idxes = self.parse_excluded_columns(exclude_column_idxes)
            ts_cols = np.array(self._df_data.columns)
            ts_excluded_cols = ts_cols[ts_excluded_cols_idxes]
            # self._df_data.drop(ts_excluded_cols, axis=1, inplace=True)
        if exclude_columns:
            self._df_data_copy = self._df_data.copy()
            ts_excluded_cols = exclude_columns.split(",")
            self._df_data.drop(ts_excluded_cols, axis=1, inplace=True)

        # columns
        self._ts_columns = list(self._df_data.columns)
        in_total_cols = len(self._ts_columns)
        self._ts_str_idxes = [
            self._ts_columns.index(st_col_name) for st_col_name in ts_str_col_names
        ]
        self._ts_str_idxes = np.array(self._ts_str_idxes)
        self._ts_num_idxes = np.setdiff1d(
            np.array(range(1, in_total_cols)), np.array(self._ts_str_idxes)
        )

        # since first column is excluded, index is reduced by one
        self._ts_str_idxes -= 1
        self._ts_num_idxes -= 1

        # target column
        self._in_target_idx = self._ts_columns.index("T (degC)") - 1
        print("INFO: index of target column (T degC): %d" % (self._in_target_idx))

        self._ar_values = self._df_data.values[:, 1:]
        in_row, in_column = self._ar_values.shape

        print("INFO: # of columns: %d and rows: %d " % (in_row, in_column))
        print("INFO: # of string cols: {}".format(len(self._ts_str_idxes)))
        print("INFO: # of integer cols: {}".format(len(self._ts_num_idxes)))

    def identify_outlier(self, sigma=3.0):
        """
    Aims: 
     - identify the outlier of each feature with numeric value 
     - three std from mean is considered as outlier since the random 
       variables follow normal distribution. Can be utilized other techniques
       such IQR and clustering.
     - update the outlier values with respective mean observed from training 
       dataset
    Parameters:
      sigma: number of sigma for outlier
    """

        in_num_sample = self._ar_values.shape[0]
        in_ubound = int(self._fo_train_ratio * in_num_sample)
        print("INFO: {} sigma was choosen for outlier".format(sigma))

        # find outlier
        for idx in self._ts_num_idxes:
            fo_mean = self._ar_values[:in_ubound, idx].mean(axis=0)
            fo_std = self._ar_values[:in_ubound, idx].std(axis=0)
            # idx+1 is for excluding date time column
            print(
                "INFO: feature: {0:s} mean: {1:.2f} std: {2:.2f}".format(
                    self._ts_columns[idx + 1], fo_mean, fo_std
                )
            )

            # Upper bound
            fo_ubound = fo_mean + (sigma * fo_std)
            ar_outlier = self._ar_values[:, idx] >= fo_ubound
            in_num_outlier = ar_outlier.sum()
            if in_num_outlier > 0:
                self._ar_values[ar_outlier, idx] = fo_mean
                print(
                    "INFO: UPPER BOUND - threshold: {0:.2f} # of outlier: {1:d}".format(
                        fo_ubound, in_num_outlier
                    )
                )

            # Lower bound
            fo_ubound = fo_mean - (sigma * fo_std)
            ar_outlier = self._ar_values[:, idx] <= fo_ubound
            in_num_outlier = ar_outlier.sum()
            if in_num_outlier > 0:
                self._ar_values[ar_outlier, idx] = fo_mean
                print(
                    "INFO: LOWER BOUND - threshold: {0:.2f} # of outlier: {1:d}".format(
                        fo_ubound, ar_outlier.sum()
                    )
                )

        print("INFO: Looked and updated outliers!!!")
        print("---")

    def fit_transform_standard_scaler(self):
        """
    Aims: 
          Standarize using training dataset and transform entire 
          dataset using mean and standard deviation observed from 
          training dataset. 
          
          It is only applied to columns with numerical values.
    """

        in_num_sample = self._ar_values.shape[0]
        in_ubound = int(self._fo_train_ratio * in_num_sample)

        # standarized the values using
        self._oj_std_scale = StandardScaler()
        self._oj_std_scale.fit(self._ar_values[:in_ubound, self._ts_num_idxes])
        ar_temp = self._oj_std_scale.transform(self._ar_values[:, self._ts_num_idxes])
        self._ar_values_og = np.copy(self._ar_values)
        self._ar_values[:, self._ts_num_idxes] = ar_temp

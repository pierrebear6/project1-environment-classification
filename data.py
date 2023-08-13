# IMPORT, CLEAN AND PREP DATA

import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
import math
from ydata_profiling import ProfileReport
from ta.volatility import AverageTrueRange
from functions.feature_stationarity import *

filepath = 'D:/simon/portfolio_projects/project1'


class GatherCandlestickData():
    def __init__(self, ticker=None, interval='1d', period='6y', output='full'):
        """
        :param ticker: stock ticker
        :param interval: time interval
        :param period: time period
        :param output: conditional - to return df output (print)
        """
        self.ticker = ticker
        self.interval = interval
        self.period = period
        self.output = output

    def import_compare_data(self):
        """
        Note: Results in compilation of stock data to check with another data source. Separate function due to API
        request limits
        :return: alpha_data: stock data from alpha vantage
        """
        # output size can also be set to 'full'
        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"function": "TIME_SERIES_DAILY", "symbol": self.ticker, "outputsize": 'full', "datatype": "json"}
        headers = {
            "X-RapidAPI-Key": "ad9bd43ad3msh85edfc8412a4837p1eb8acjsn625bfe4be209",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        json = response.json()
        alpha_data = pd.DataFrame(json['Time Series (Daily)']).transpose()
        alpha_data = alpha_data.iloc[::-1]
        date_trim = int(self.period[0]) * 252
        alpha_data = alpha_data[-date_trim:]
        alpha_data.reset_index(inplace=True)
        alpha_data = alpha_data.rename(columns={'index': 'Date'})
        alpha_data.rename(
            columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'},
            inplace=True)
        alpha_data[['Open', 'High', 'Low', 'Volume', 'Close']] = alpha_data[
            ['Open', 'High', 'Low', 'Volume', 'Close']].apply(
            pd.to_numeric)

        def roundup(number):
            return math.ceil(number * 100) / 100

        alpha_data[['Open', 'High', 'Low', 'Close']] = alpha_data[['Open', 'High', 'Low', 'Close']].apply(
            lambda x: x.apply(roundup))

        alpha_data.to_csv(f'{filepath}/data/imports/{self.ticker}_alpha.csv', index=False)
        return alpha_data

    def import_data(self):
        """
        :return: yfin_data: stock data from yfinance
        """
        # Store data or just use this function? Check later

        #
        yfin_data = yf.Ticker(self.ticker).history(period=self.period, interval=self.interval)
        yfin_data = yfin_data
        date_trim = int(self.period[0]) * 252
        yfin_data = yfin_data[-date_trim:]
        yfin_data.reset_index(inplace=True)
        yfin_data['Date'] = pd.to_datetime(yfin_data['Date']).dt.strftime('%Y-%m-%d')
        yfin_data = yfin_data.drop(['Stock Splits', 'Dividends'], axis=1)

        def roundup(number):
            return math.ceil(number * 100) / 100

        yfin_data[['Open', 'High', 'Low', 'Close']] = yfin_data[['Open', 'High', 'Low', 'Close']].apply(
            lambda x: x.apply(roundup))
        yfin_data.reset_index(drop=True)

        return yfin_data


class PrepareData():
    def __init__(self, ticker, df1, tolerance=.05, period='6y', replace_zero=False):
        """
        :param ticker: stock symbol
        :param df1: primary df
        :param tolerance: tolerance for comparison error between datasets
        :param period: length of historical data
        :param replace_zero: bool variable to determine if zero values should be imputed
        """
        self.df1 = df1
        self.tolerance = tolerance
        self.period = period
        self.ticker = ticker
        self.replace_zero = replace_zero

    def compare_data(self, df2):
        """
        :param df2: secondary df used to compare primary df with
        :return: Feedback on if the data sources are consistent
        """
        df1 = self.df1[['Open', 'High', 'Low', 'Close', 'Volume']]
        df2 = df2[['Open', 'High', 'Low', 'Close', 'Volume']]
        tolerance = self.tolerance
        # Check if the DataFrames have the same shape
        if df1.shape != df2.shape:
            print('Dataframe shapes do not match!')
            return False

        diff = np.abs(df1 - df2)
        max_tolerance = np.max(np.abs(df1) * tolerance)

        # Check if all values are similar within the tolerance
        are_similar = np.all(diff <= max_tolerance)

        # Return the rows where DataFrames are not similar
        if not are_similar:
            percent_diff = (diff / df1) * 100
            rows_diff = percent_diff[~np.all(diff <= max_tolerance, axis=1)]
            # print(rows_diff)
            print('Data source not consistent!')

            return False
        else:
            print('Data source consistent!')
            return True

    def clean_data(self):
        """
        :return: Either a clean df or feedback on why data is faulty
        """
        df = self.df1
        period = int(self.period[0])
        ticker = self.ticker
        replace_zero = self.replace_zero

        try:
            fill = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in fill:
                if replace_zero:
                    mask = df[col] == 0
                    df[col] = df[col].where(~mask, (df[col].shift(-1) + df[col].shift(1)) / 2)
            data_small = df.shape[0] < period * 251
            nan_present = df.isna().values.any()
            zero_present = (df == 0).any().any()
            empty_df = df.empty

            text_map = {data_small: 'The dataset is too small.', nan_present: 'NaN/Null values are present',
                        zero_present: 'Zeroes are present', empty_df: 'The dataset is empty'}
            dirty = False
            for var, text in text_map.items():
                if var:
                    print(text)
                    dirty = True
            if not dirty:
                df.to_csv(rf'{filepath}\data\clean_data\{ticker}_clean.csv', index=False)
                print('Clean data stored!')
                return df
        except KeyError as e:
            error_message = str(e)
            if "1d data not available" or "not found in axis" in error_message:
                print("Data not available for the specified time range. Skipping operation.")
            else:
                print("An error occurred:", error_message)

    def split_data(self, train_len):
        """
        :param train_len: Proportion of data that is assigned to the training dataset
        :return: train and test dfs
        """
        n = len(self.df1)
        df = self.df1
        train_df = df[0:int(n * train_len)]
        test_df = df[int(n * train_len):]
        return train_df, test_df

    def minmaxscalar(self):
        """
        :return: Minmax scaled df
        """
        data = self.df1
        data = data.drop(['Date'], axis=1)
        col_names = data.columns
        scaler = MinMaxScaler().fit(data.values)
        scaled_data_np = scaler.transform(data.values)
        scaled_data_df = pd.DataFrame(scaled_data_np, columns=col_names)
        return scaled_data_df

    def inverse_scale_data(self, scaled_data):
        """
        :param scaled_data: Minmax scaled data
        :return: inverse scaled data
        """
        data = self.df1
        col_names = scaled_data.columns
        scaler = MinMaxScaler().fit(data.values)
        inverse_scaled_np = scaler.inverse_transform(scaled_data)
        inverse_scaled_df = pd.DataFrame(inverse_scaled_np, columns=col_names)
        return inverse_scaled_df

    def xy_split(self, target, train_len):
        """
        :param target: Target column to be used as label
        :param train_len: Proportion of data that is assigned to the training dataset
        :return: Train and test splits
        """
        df = self.df1
        n = len(self.df1)
        train_df = df[0:int(n * train_len)]
        test_df = df[int(n * train_len):]
        train_df = train_df.sample(frac=1)
        test_df = test_df.sample(frac=1)
        X_train = train_df.drop(target, axis=1).values
        y_train = train_df[target].values
        X_test = test_df.drop(target, axis=1).values
        y_test = test_df[target].values
        return X_train, y_train, X_test, y_test


class AnalyzeData():
    def __init__(self, df, data_name):
        self.df = df
        self.data_name = data_name

    def get_pd_sumstats(self):
        """
        :return: None. Prints basic pandas description statistics
        """
        df = self.df
        print(df.describe())

    def get_ydata_sumstats(self):
        """
        :return: Statistical report on df
        """
        df = self.df
        data_name = self.data_name
        profile = ProfileReport(df, title=data_name)
        profile.to_file(rf'{filepath}\data\summary_stats\{data_name}_stats_report.html')
        return df


def add_features(df, window=15):
    """
    :param df: Df to be used to add features
    :param window: Used for rolling window calculations or time horizon calculations
    :return: Df with added features
    """
    df['Volatility'] = (df['Close'].pct_change().rolling(252).std() * (252 ** 0.5) * 100) / df['Close']
    df['Volatility_Pct'] = ((df['Volatility'] - df['Volatility'].rolling(window=window).min()) /
                            (df['Volatility'].rolling(window=window).max() - df['Volatility'].rolling(
                                window=window).min()))
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window).average_true_range()
    df['ATR_15'] = abs(df['Close'] - df['Close'].shift(-window))
    df['15Diff'] = df['Close'].shift(-window) - df['Close']
    df['ATR_Diff'] = abs(df['ATR_15'] - df['ATR'])
    df['ATR_AvgDiff'] = df['ATR_Diff'].rolling(window=window).mean()
    df['Target'] = 0
    df.loc[(df['ATR_Diff'] > df['ATR_AvgDiff']) & (df['15Diff'] > 0), 'Target'] = 1
    df.loc[(df['ATR_Diff'] > df['ATR_AvgDiff']) & (df['15Diff'] < 0), 'Target'] = -1
    df = df.copy()[0:-window]
    df['LogReturn'] = np.log10(df['Close'] / df['Close'].shift())
    df['varLogReturn'] = df['LogReturn'].rolling(window=window).var()
    df['stdLogReturn'] = df['LogReturn'].rolling(window=window).std()
    df['Return'] = df['Close'].pct_change()
    df['varReturn'] = df['Return'].rolling(window=window).var()
    df['stdReturn'] = df['Return'].rolling(window=window).std()
    df['NightGain'] = df['Open'] / df['Close'].shift() - 1
    df['IntradayGain'] = df['Close'] / df['Open'] - 1
    df = df.dropna()

    ffd_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'LogReturn', 'NightGain', 'IntradayGain']
    for ffd_col in ffd_columns:
        series = df[[ffd_col]].copy()
        df[f'ffd_adf_{ffd_col}'] = find_stat_series(df=series)
    dec_ts_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for dec_ts_col in dec_ts_columns:
        df[f'residual_{dec_ts_col}'] = decompose_time_series(df=df, column_name=dec_ts_col)

    df = df.dropna()
    df = df.drop(['ATR_15', '15Diff', 'ATR_Diff', 'ATR_AvgDiff'], axis=1)
    df = df.reset_index(drop=True)
    return df

# start with list of stocks by category -> just run thru lists w clean_data folder -> one shot compile cumulative
# data -> feed into model

# INFERENTIAL TEST OF DATA EXAMPLE

from data import *
time_range = 15


def inference_features(df, window):
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

    df['ATR_Diff'] = df['ATR_15'] - df['ATR']
    df['ATR_AvgDiff'] = df['ATR_Diff'].rolling(window=window).mean()

    df = df.dropna()
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', ], axis=1)
    return df


amd = GatherCandlestickData(ticker='amd', period='5y')
amd_df = amd.import_data()
amd_df_features = inference_features(df=amd_df, window=time_range)
print(amd_df_features)

analyze_data = AnalyzeData(df=amd_df_features, data_name='amd_inferential_test')
profile = analyze_data.get_ydata_sumstats()

# NO LINEAR CORRELATIONS - refer to ydata report

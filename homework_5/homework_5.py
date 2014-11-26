import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da

"""
Bollinger band generation

Takes look back for a stock and generates:

Rolling mean
Stock price
Upper band
Lower band

Traditionally two standard deviations are used, not one
"""

def get_stock_data(dt_start, dt_end, ls_syms):
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_syms, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    return d_data, ldt_timestamps


def generate_bollinger_bands(ls_sym, lookback, dt_start, dt_end):
    df_data, ldt_timestamps = get_stock_data(dt_start, dt_end, ls_sym)

    df_close = df_data['close']

    df_rolling_mean = pd.rolling_mean(df_close, lookback)
    df_rolling_std_dev = pd.rolling_std(df_close, lookback)
    df_upper_band = df_rolling_mean + df_rolling_std_dev
    df_lower_band = df_rolling_mean - df_rolling_std_dev

    df_bollinger = (df_close - df_rolling_mean) / (df_rolling_std_dev)

    df_bollinger = df_bollinger.fillna(0)

    # Determine where to put vertical lines (bollinger band crossings)
    na_bollinger_prev = np.roll(df_bollinger.values, 1, axis=0)
    na_bollinger_prev[0] = np.NaN
    na_upper=np.logical_or(np.logical_and(df_bollinger.values>=1.0, na_bollinger_prev<1.0),
                           np.logical_and(df_bollinger.values<1.0, na_bollinger_prev>=1.0))
    na_lower=np.logical_or(np.logical_and(df_bollinger.values<=-1.0, na_bollinger_prev>-1.0),
                           np.logical_and(df_bollinger.values>-1.0, na_bollinger_prev<=-1.0))
    df_upper = pd.DataFrame(na_upper, index=df_bollinger.index, columns=df_bollinger.columns)
    df_lower = pd.DataFrame(na_lower, index=df_bollinger.index, columns=df_bollinger.columns)

    for symbol in ls_sym:

        plt.clf()

        # Chart 1
        plt.subplot(211)
        df_close[symbol].plot(color='b')
        df_rolling_mean[symbol].plot(color='r')
        plt.fill_between(ldt_timestamps, df_lower_band[symbol], df_upper_band[symbol], 
            alpha=0.15, color='red')

        for ts in df_upper[symbol].index:
            if df_upper[symbol].ix[ts]:
                plt.axvline(x=ts, color='red')
            if df_lower[symbol].ix[ts]:
                plt.axvline(x=ts, color='green')

        plt.legend(['Adj Close', str(lookback) + ' day MA'], loc=4)
        plt.ylabel('Price')

        # Chart 2
        plt.subplot(212)
        df_bollinger[symbol].plot()
        plt.fill_between(ldt_timestamps, -1.0, 1.0, alpha=0.15, color='red')
        
        for ts in df_upper[symbol].index:
            if df_upper[symbol].ix[ts]:
                plt.axvline(x=ts, color='red')
            if df_lower[symbol].ix[ts]:
                plt.axvline(x=ts, color='green')

        plt.legend(['Bollinger'], loc=4)
        plt.ylabel('Bollinger Feature')

        
        # Alternative implementation
        """for i in range(1, df_bollinger[symbol].size):
            prev = df_bollinger[symbol][i-1]
            current = df_bollinger[symbol][i]
            ts_prev = df_bollinger[symbol].index[i-1]
            ts_current = df_bollinger[symbol].index[i]

            if prev <= 1.0 and current > 1.0:
                plt.axvline(x=ts_current, color='red')
            elif prev > 1.0 and current <= 1.0:
                plt.axvline(x=ts_prev, color='red')
            elif prev <= -1.0 and current > -1.0:
                plt.axvline(x=ts_current, color='green')
            elif prev > -1.0 and current <= -1.0:
                plt.axvline(x=ts_prev, color='green')"""

        plt.tight_layout()
        plt.savefig(symbol + '_' + str(lookback) + '_day_bollinger.pdf', 
            format='pdf')

    pd.set_option('display.precision', 7)
    print(df_bollinger.to_string())


if __name__ == '__main__':
    ls_sym = ['AAPL', 'GOOG', 'IBM', 'MSFT']
    lookback = 20
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)

    generate_bollinger_bands(ls_sym, lookback, dt_start, dt_end)
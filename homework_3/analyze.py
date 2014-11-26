import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

def main():
    if len(sys.argv) is not 3:
        raise "Usage: " + sys.argv[0] + " <values>.csv <benchmark>"

    values_file = sys.argv[1]
    benchmark_sym = sys.argv[2]

    dt_timeofday = dt.timedelta(hours=16)

    # Obtain normalised price
    parse = lambda x: dt.datetime.strptime(x, '%Y %m %d') + dt_timeofday
    df_values = pd.read_csv(values_file, 
        names=['year', 'month', 'day', 'value'], 
        parse_dates=[['year', 'month', 'day']],
        date_parser=parse,
        index_col=0)

    na_port_values = df_values.values

    dt_start = df_values.index[0].to_datetime()
    dt_end = df_values.index[-1].to_datetime()

    d_data = get_stock_data(dt_start, dt_end, [benchmark_sym])
    # Getting the numpy ndarray of close prices.
    na_bm_price = d_data['close'].values
    

    port_sharpe_ratio, port_total_returns, port_std_dev, port_mean = get_performance(na_port_values)
    bm_sharpe_ratio, bm_total_returns, bm_std_dev, bm_mean = get_performance(na_bm_price)


    print("The final value of the portfolio using the sample file is -- " +\
        str(df_values.index[-1].date()) + " " + str(df_values.ix[-1][0]))

    print("Details of the Performance of the portfolio :")

    print("Data Range: " + str(dt_start) + " to " + str(dt_end))

    print("Sharpe Ratio of Fund : ", port_sharpe_ratio)
    print("Sharpe Ratio of " + benchmark_sym + " : ", bm_sharpe_ratio)

    print("Total Return of Fund :  ", port_total_returns)
    print("Total Return of " + benchmark_sym + " : ", bm_total_returns)

    print("Standard Deviation of Fund : ", port_std_dev)
    print("Standard Deviation of " + benchmark_sym + " : ", bm_std_dev)

    print("Average Daily Return of Fund : ", port_mean)
    print("Average Daily Return of " + benchmark_sym + " : ", bm_mean)

    # Generate chart
    na_bm_price = na_bm_price * (1 / (na_bm_price[0, :] / na_port_values[0, :]))

    generate_chart(df_values.index, 'Portfolio_' + values_file, 
        benchmark_sym, na_port_values, na_bm_price)


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

    return d_data


def get_performance(na_price):
    na_normalized_price = na_price / na_price[0.0, :]

    # Our daily return
    tsu.returnize0(na_normalized_price)

    sharpe_ratio = tsu.get_sharpe_ratio(na_normalized_price)[0]
    na_cumulative_returns = np.cumprod(na_normalized_price + 1)
    total_returns = na_cumulative_returns[-1]
    std_dev = np.std(na_normalized_price)
    mean = np.mean(na_normalized_price)

    return sharpe_ratio, total_returns, std_dev, mean


def generate_chart(ldt_timestamps, label1, label2, na_rets1, na_rets2):
    
    # Plotting the plot of daily returns
    plt.clf()
    plt.plot(ldt_timestamps, na_rets1)
    plt.plot(ldt_timestamps, na_rets2)
    plt.axhline(y=0, color='r')
    plt.legend([label1, label2], loc=4)
    plt.ylabel('Daily Returns')
    plt.xlabel('Date')
    plt.savefig(label1 + '.pdf', format='pdf')


if __name__ == '__main__':
    main()
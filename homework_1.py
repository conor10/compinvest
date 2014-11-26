import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

def main():
    test1()
    test2()
    test3()
    homework1()

def test1():
    print("Test 1")
    ls_syms = ['AAPL', 'GLD', 'GOOG', 'XOM']
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    ls_alloc = [0.4, 0.4, 0.0, 0.2]
    test(dt_start, dt_end, ls_syms, ls_alloc)

def test2():
    print("Test 2")
    ls_syms = ['AXP', 'HPQ', 'IBM', 'HNZ']
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)
    ls_alloc = [0.0, 0.0, 0.0, 1.0]
    test(dt_start, dt_end, ls_syms, ls_alloc)

def test3():
    print("Test 3")
    ls_syms = ['AAPL', 'GOOG', 'XOM', 'GLD']
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    ls_alloc = [0.4, 0.0, 0.2, 0.4]
    test(dt_start, dt_end, ls_syms, ls_alloc)

def test(dt_start, dt_end, ls_syms, ls_alloc):

    print("Start date: ", dt_start.strftime('%B %d, %Y'))
    print("End date: ", dt_end.strftime('%B %d, %Y'))
    print("Symbols: ", ls_syms)
    print("Optimal allocations: ", ls_alloc)

    na_returns_alloc, sharpe_ratio, std_dev, mean, cum_returns = simulate(dt_start, dt_end, ls_syms, ls_alloc)

    print("Sharpe ratio: ", sharpe_ratio)
    print("Volatility (stdev of daily returns): ", std_dev)
    print("Average daily returns", mean)
    print("Cumulative returns: ", cum_returns)

def homework1():
    question1()
    question2()

def question1():
    print("Question 1")
    ls_syms = ['C', 'GS', 'IBM', 'HNZ']
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    find_optimal(dt_start, dt_end, ls_syms)

def question2():
    print("Question 2")
    ls_syms = ['BRCM', 'TXN', 'AMD', 'ADI']
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    find_optimal(dt_start, dt_end, ls_syms)

def find_optimal(dt_start, dt_end, ls_syms):

    max_sharpe = 0
    ls_optimal_alloc = []
    na_opt_returns_alloc = None

    for v1 in np.arange(0.0, 1.1, 0.1):
        for v2 in np.arange(0.0, 1.1, 0.1):
            for v3 in np.arange(0.0, 1.1, 0.1):
                for v4 in np.arange(0.0, 1.1, 0.1):
                    if v1 + v2 + v3 + v4 == 1:
                        ls_alloc = [v1, v2, v3, v4]
                        na_returns_alloc, sharpe_ratio, std_dev, mean, cum_returns = simulate(dt_start, dt_end, ls_syms, ls_alloc)
                        #print("Checking allocation: ", ls_alloc)
                        if sharpe_ratio > max_sharpe:
                            max_sharpe = sharpe_ratio
                            ls_optimal_alloc = ls_alloc
                            na_opt_returns_alloc = na_returns_alloc

    print("Symbols: ", ls_syms)
    print("Optimal allocation: ", ls_optimal_alloc)
    print("Sharpe ratio: ", max_sharpe)

    generate_versus_spx(dt_start, dt_end, ls_syms, ls_optimal_alloc, na_opt_returns_alloc)


def simulate(dt_start, dt_end, ls_syms, ls_alloc):

    """
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
    """
    d_data = get_stock_data(dt_start, dt_end, ls_syms)
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values

    na_normalized_price = na_price / na_price[0, :]

    # Daily returns by allocation
    na_returns_alloc = np.sum(na_normalized_price * ls_alloc, axis=1)

    na_rets = na_returns_alloc.copy()
    # Our daily return
    tsu.returnize0(na_returns_alloc)

    sharpe_ratio = tsu.get_sharpe_ratio(na_returns_alloc)
    na_cumulative_returns = np.cumprod(na_returns_alloc + 1)
    std_dev = np.std(na_returns_alloc)
    mean = np.mean(na_returns_alloc)

    return na_rets, sharpe_ratio, std_dev, mean, na_cumulative_returns[-1]


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

def generate_versus_spx(dt_start, dt_end, ls_syms, ls_optimal_alloc, portfolio_returns):
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    d_data = get_stock_data(dt_start, dt_end, ['$SPX'])
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values
    na_normalized_price = na_price / na_price[0, :]
    
    name = 'portfolio (' + ','.join([k + ':' + str(v) for k, v in zip(ls_syms, ls_optimal_alloc)]) + ')'

    generate_chart(ldt_timestamps, '$SPX', name, na_normalized_price, portfolio_returns)

def generate_chart(ldt_timestamps, label1, label2, na_rets1, na_rets2):
    
    # Plotting the plot of daily returns
    plt.clf()
    plt.plot(ldt_timestamps, na_rets1)
    plt.plot(ldt_timestamps, na_rets2)
    plt.axhline(y=0, color='r')
    plt.legend([label1, label2], loc=4)
    plt.ylabel('Daily Returns')
    plt.xlabel('Date')
    plt.savefig(label2 + '.pdf', format='pdf')



if __name__ == '__main__':
    main()
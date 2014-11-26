import csv
import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""


def find_events(ls_symbols, d_data, close_price):
    ''' Finding the event dataframe '''
    df_close = d_data['actual_close']
    ts_market = df_close['SPY']

    print "Finding Events"

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    ls_orders = []

    # Time stamps for the event range
    ldt_timestamps = df_close.index

    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]

            # Event is found if the symbol drops below $5.0
            if f_symprice_yest >= close_price and f_symprice_today < close_price:
                # Signal has been hit
                df_events[s_sym].ix[ldt_timestamps[i]] = 1

                ts_start = ldt_timestamps[i]
                if i + 5 < len(ldt_timestamps):
                    ts_end = ldt_timestamps[i + 5]
                else:
                    ts_end = ldt_timestamps[-1]
                
                ls_orders.append([ts_start.year, ts_start.month, ts_start.day, s_sym, 'Buy', 100])
                ls_orders.append([ts_end.year, ts_end.month, ts_end.day, s_sym, 'Sell', 100])
                
    ls_orders.sort() # Returns None so must call here
    return df_events, ls_orders

def run_study(dt_start, dt_end, symbol_list, name, close_price):
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list(symbol_list)
    ls_symbols.append('SPY')

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_events, ls_orders = find_events(ls_symbols, d_data, close_price)
    print "Creating Study"
    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                s_filename=name + '.pdf', b_market_neutral=True, b_errorbars=True,
                s_market_sym='SPY')

    print "Writing orders"

    with open(name + '_orders.csv', 'w') as f:
        writer = csv.writer(f)
        for order in ls_orders:
            writer.writerow(order)


if __name__ == '__main__':
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    #run_study(dt_start, dt_end, 'sp5002012', 'S&P500_2012', 5.0)
    run_study(dt_start, dt_end, 'sp5002012', 'S&P500_2012_Q1_9_0', 9.0)
    run_study(dt_start, dt_end, 'sp5002012', 'S&P500_2012_Q2_6_0', 6.0)


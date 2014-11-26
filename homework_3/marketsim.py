import csv
import datetime as dt
import numpy as np
import pandas as pd
import sys

import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du

def main():
    if len(sys.argv) is not 4:
        raise "Usage: " + sys.argv[0] + " <cash> <orders>.csv <values>.csv"

    cash = int(sys.argv[1])
    orders_file = sys.argv[2]
    values_file = sys.argv[3]

    parse = lambda x: dt.datetime.strptime(x, '%Y %m %d')
    df_orders = pd.read_csv(orders_file, 
        #names=['year', 'month', 'day', 'symbol', 'side', 'price', 'na'], 
        names=['year', 'month', 'day', 'symbol', 'side', 'price'], 
        parse_dates=[['year', 'month', 'day']],
        date_parser=parse,
        index_col=0)
    na_orders = df_orders.values

    ls_symbols = list(set(na_orders[:, 0]))

    dt_start = df_orders.index[0].to_datetime()
    dt_end = df_orders.index[-1].to_datetime() + dt.timedelta(days=1)

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    ls_order_ts = []
    for ts in df_orders.index:
        ls_order_ts.append(ts.to_datetime() + dt_timeofday)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    # Copying close price into separate dataframe to find rets
    df_close = d_data['close'].copy()

    # Create our holdings df
    df_trade = pd.DataFrame(index=ldt_timestamps, columns=ls_symbols, dtype='float64')
    df_trade = df_trade.fillna(0.0)

    s_cash = pd.Series(0.0, index=ldt_timestamps)
    cash_remaining = cash

    s_positions = pd.Series(0.0,  index=ls_symbols)


    # Suggested implementation

    s_cash[0] = cash

    for i in range(0, len(df_orders.index)):
        order = df_orders.ix[i]
        ts = df_orders.index[i] + dt_timeofday

        symbol = order[0]
        side = order[1]
        qty = order[2]
        if side == 'Sell':
            qty = -qty

        df_trade.ix[ts][symbol] += qty

        price = qty * df_close.ix[ts][symbol]
        s_cash.ix[ts] -= price

    df_close['_CASH'] = 1.0
    df_trade['_CASH'] = s_cash

    # Cumulative sum of holdings accross each column
    df_holdings = df_trade.cumsum(axis=0)

    # Cumulative sum of holdings accross each row
    df_daily_value = (df_holdings * df_close).cumsum(axis=1)

    with open(values_file, 'w') as f:
        writer = csv.writer(f)
        for ts in df_daily_value.index:
            value = df_daily_value.ix[ts]['_CASH']
            writer.writerow((ts.year, ts.month, ts.day, value))


    # My implementation - still correct
    """
    # Calculate holdings in each stock each day
    for i in range(0, len(ldt_timestamps)):
        ts = ldt_timestamps[i]
        # If we place an order on this date
        # Would good to just use df_orders
        if ts in ls_order_ts:
            orders = df_orders.ix[ts.date()]
            # Series - there must be a better way to identify this
            if orders.values.size is len(orders):
                symbol = orders[0]
                side = orders[1]
                qty = orders[2]
                if side == 'Sell':
                    qty = -qty
                # Update holdings for symbol
                s_positions[symbol] += qty
                # Update costs
                price = qty * df_rets.ix[ts][symbol]
                cash_remaining -= price
            # DataFrame
            else:
                for i in range(0, len(orders)):
                    symbol = orders.ix[i][0]
                    side = orders.ix[i][1]
                    qty = orders.ix[i][2]
                    if side == 'Sell':
                        qty = -qty
                    
                    # Update holdings for symbol
                    s_positions[symbol] += qty
                    # Update costs
                    price = qty * df_rets.ix[ts][symbol]
                    cash_remaining -= price

        df_holdings.ix[ts] = s_positions
        s_cash.ix[ts] = cash_remaining

    # Calculate value of holdings each day
    df_value = df_holdings * df_rets

    # Total fund value each day
    df_total_value = np.sum(df_value.values, axis=1) + s_cash

    # Write results out to csv file
    with open(values_file, 'w') as f:
        writer = csv.writer(f)
        idx_total = df_total_value.index
        for i in range(0, len(df_total_value)):
            ts = idx_total[i]
            value = df_total_value[i]
            #value = int(round(df_total_value[i]))
            writer.writerow((ts.year, ts.month, ts.day, value))
    """
    # End

if __name__ == '__main__':
    main()
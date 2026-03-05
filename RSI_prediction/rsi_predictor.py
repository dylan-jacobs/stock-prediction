# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:47:46 2022

@author: dylan
"""
import yfinance
import numpy as np
import os
import pandas as pd
import backtesting
import multiprocessing
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps

HISTORY = '60d'
INTERVAL = '15m' # 1h: max 2y
TICKER = 'SPY'

def load_close_data(ticker, period):
    data = yfinance.download(ticker, period=period, interval=INTERVAL)
    close_data = data[['Close']]
    open_data = data[['Open']]
    date_data = data.index.to_frame(index=False, name='Date')  # Extract Date from the index
    high_data = data[['High']]
    low_data = data[['Low']]
    volume = data[['Volume']]
    return open_data, close_data, date_data, high_data, low_data, volume
    
def calculate_rsi(closes, window_len):
    gains = []
    losses = []
    window = []
    prev_avg_gain = None
    prev_avg_loss = None
    close_data = [float(i) for i in closes.values]
    rsi_vals = []
    for i, close in enumerate(close_data):
        gain = 0
        loss = 0
        if i == 0:
            window.append(close)
            rsi_vals.append(None) # this will get removed later - it's just so that initial dimensions will match
            continue
        
        dif = close_data[i] - close_data[i-1]
        
        if dif > 0:
            gain = dif
            loss = 0
            
        elif dif < 0:
            gain = 0
            loss = abs(dif)
        
        gains.append(gain)
        losses.append(loss)
        
        if i < window_len:
            window.append(close)
            rsi_vals.append(None)
            continue
        
        if i == window_len:
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(gains)
        
        else:
            avg_gain = (prev_avg_gain * (window_len - 1) + gain) / window_len
            avg_loss = (prev_avg_loss * (window_len - 1) + loss) / window_len
        
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss
        
        rs = avg_gain / avg_loss
        rsi = (100 - (100 / (1 + rs)))
        
        window.append(close)
        window.pop(0)
        gains.pop(0)
        losses.pop(0)
        
        rsi_vals.append(rsi)
    output = np.asarray(rsi_vals)
    return output


def backtest_strategy(df, predictions):

    class RSIStrategy(Strategy):

        buy_threshold = 0.3
        sell_threshold = 0.7
        pred_data = None

        def init(self):
            self.rsi = self.I(lambda: self.pred_data, name='RSI Predictions')

        def next(self):
            rsi = self.rsi[-1]
            
            if rsi < self.buy_threshold*100:
                if not self.position.is_long:
                    self.buy()
            elif rsi > self.sell_threshold*100:
                if self.position.is_long:
                    self.position.close()

    split = int(0.8 * len(df))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    RSIStrategy.pred_data = predictions[:split].flatten()
    train_bt = Backtest(train_df, RSIStrategy, cash=10000, commission=0, finalize_trades=True)
    buy_thresholds = [float(round(i, 2)) for i in np.arange(0.25, 0.35, 0.01)]
    sell_thresholds = [float(round(i, 2)) for i in np.arange(0.55, 0.85, 0.01)]
    stats, heatmap = train_bt.optimize(
        buy_threshold=buy_thresholds,
        sell_threshold=sell_thresholds, 
        constraint=lambda p: p.buy_threshold < p.sell_threshold, # buy threshold must be less than sell threshold
        maximize='Return [%]',
        max_tries=None,
        random_state=42,
        return_heatmap=True,
        return_optimization=False
    )
    plot_heatmaps(heatmap, agg='mean')
    #train_bt.plot()

    # get optimal parameters
    optimal_buy_threshold = stats.at['_strategy'].buy_threshold
    optimal_sell_threshold = stats.at['_strategy'].sell_threshold
    print(f'Best buy threshold: {optimal_buy_threshold}, Best sell threshold: {optimal_sell_threshold}')
    
    RSIStrategy.buy_threshold = optimal_buy_threshold
    RSIStrategy.sell_threshold = optimal_sell_threshold
    RSIStrategy.pred_data = predictions[split:].flatten()
    test_bt = Backtest(test_df, RSIStrategy, cash=10000, commission=0, finalize_trades=True)
    stats = test_bt.run()
    print(stats)
    #test_bt.plot()

    return optimal_buy_threshold, optimal_sell_threshold

def test_rsi_strategy(ticker=TICKER):
    close_col = 'Close'
    # get close data
    open_data, close_data, dates, highs, lows, volume = load_close_data(ticker, HISTORY)
    data = pd.DataFrame(close_data.values, columns=[close_col])

    data['Date'] = dates.values.ravel()
    data['Open'] = open_data.values
    data['High'] = highs.values
    data['Low'] = lows.values
    data['Volume'] = volume.values
    
    # add rsi
    rsi = calculate_rsi(close_data, 14)
    data['rsi'] = rsi

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
        
    # extract dates
    dates = data['Date']
    data = data.drop(columns=['Date']) # remove dates

    close_data = data[close_col]

    # backtest
    df = pd.DataFrame({
        'Open': data['Open'].values.flatten(),
        'High': data["High"].values.flatten(),
        'Low': data["Low"].values.flatten(),
        'Close': data['Close'].values.flatten(),
        'Volume': data["Volume"].values.flatten()
    }, index=pd.DatetimeIndex(dates.values.flatten()))
    optimal_buy_threshold, optimal_sell_threshold = backtest_strategy(df, data['rsi'].values.flatten())

    return rsi[-1], optimal_buy_threshold, optimal_sell_threshold


def main():
    #backtesting.Pool = multiprocessing.Pool

    test_rsi_strategy()
    pass

if __name__=='__main__':
    main()

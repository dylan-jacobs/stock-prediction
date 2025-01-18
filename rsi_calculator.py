# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:55:17 2022

@author: dylan
"""

import yfinance
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculateRSI(ticker, period, interval):
    data = yfinance.download(ticker, period=period, interval=interval)
    data = data[['Close']]
    close_data = [float(i) for i in data.values]
    window_len = 14
    gains = []
    losses = []
    window = []
    prev_avg_gain = None
    prev_avg_loss = None
    output = [['date', 'close', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rsi']]
    
    for i, close in enumerate(close_data):
        gain = 0
        loss = 0
        if i == 0:
            window.append(close)
            output.append([i+1, close, 0, 0, 0, 0, 0])
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
            output.append([i+1, close, gain, loss, 0, 0, 0])
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
        
        output.append([i+1, close, gain, loss, avg_gain, avg_loss, rsi])
    output = np.asarray(output)
    return output[:, 6]

def test():
    rsi_buy_threshold = 40
    rsi_sell_threshold = 60
    ticker = 'SPY'
    period = '10y'
    interval = '1d'
    data = calculateRSI(ticker, period, interval)
    data = pd.DataFrame(np.column_stack([data[1:, 1], data[1:, -1]]), columns=['Close', 'rsi'])
    data = data.apply(pd.to_numeric, errors='coerce')
    data['bnh_returns'] = np.log(data['Close']/data['Close'].shift(1)) # log returns to enable 'continuous' compounding
    data = data.dropna()
    print(data.head())
    print(data.tail())

    data['signal'] = 0
    data.loc[data['rsi'] < rsi_buy_threshold, 'signal'] = 1 
    data.loc[data['rsi'] >= rsi_sell_threshold, 'signal'] = -1

    """data["moving_average_20"] = data["Close"].rolling(20).mean()

    data["signal"] = np.where(data["Close"] > data["moving_average_20"], 1, 0)
    data["signal"] = np.where(data["Close"] < data["moving_average_20"], -1, data["signal"])

    """

    data["strategy_returns"] = data["signal"].shift(1) * data["bnh_returns"]
    data["strategy_returns"] = data["strategy_returns"].fillna(0)
    data["accum_buy_and_hold"] = data["bnh_returns"].cumsum().apply(np.exp)
    data["accum_strategy_returns"] = data["strategy_returns"].cumsum().apply(np.exp)

    bnh = data['accum_buy_and_hold']
    strategy = data['accum_strategy_returns']
    plt.plot(bnh, label='Buy and Hold')
    plt.plot(strategy, label='Strategy')
    plt.legend()
    plt.show()
    print(f'Buy and Hold: {bnh.iloc[-1]}, Strategy: {strategy.iloc[-1]}')
    return data["accum_buy_and_hold"], data["accum_strategy_returns"] # return ultimate profits
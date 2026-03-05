import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import yfinance as yf
import datetime as dt
import os
import warnings
import yfinance as yfinance

def test_strategy(pred, true_closes, graph=False):
    data = pd.DataFrame(true_closes, columns=['Close'])
    data['bnh_returns'] = np.log(data['Close']/data['Close'].shift(1))

    data['predictions'] = pred
    data['signal'] = 0
    data.loc[data['predictions'] > 1*(data['Close']), 'signal'] = 1 
    data.loc[data['predictions'] < data['Close'], 'signal'] = -1
    data.dropna(inplace=True)
    data["strategy_returns"] = data["signal"].shift(1) * data["bnh_returns"]
    data["strategy_returns"] = data["strategy_returns"].fillna(0)
    data["accum_buy_and_hold"] = data["bnh_returns"].cumsum().apply(np.exp)
    data["accum_strategy_returns"] = data["strategy_returns"].cumsum().apply(np.exp)

    # plot returns against buy and hold
    bnh = data['accum_buy_and_hold']
    strategy = data['accum_strategy_returns']

    metrics = pd.DataFrame({
    'Buy & Hold': {
        'Total Return':    f"{(bnh.iloc[-1] - 1) * 100:.2f}%",
        'Volatility':      f"{bnh.std() * np.sqrt(252) * 100:.2f}%",
        'Win Rate':        f"{(bnh > 0).mean() * 100:.2f}%"
    },
    'Algorithm': {
        'Total Return':    f"{(strategy.iloc[-1] - 1) * 100:.2f}%",
        'Volatility':      f"{strategy.std() * np.sqrt(252) * 100:.2f}%",
        'Win Rate':        f"{(strategy > 0).mean() * 100:.2f}%"
    }
    })

    print(metrics)

    if graph:
        plt.plot(bnh, label='Buy and Hold')
        plt.plot(strategy, label='Strategy')
        plt.legend()
        plt.show()
        
    return data["accum_buy_and_hold"], data["accum_strategy_returns"] # return ultimate profits
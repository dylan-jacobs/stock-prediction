import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import os
import warnings
import yfinance as yfinance

def test_strategy(testX, model, input_scaler, output_scaler, closes_ind, graph=False):
    data = pd.DataFrame(input_scaler.inverse_transform(testX)[:, closes_ind], columns=['Close'])
    data['bnh_returns'] = np.log(data['Close']/data['Close'].shift(1))
    pred = model.predict(testX)
    # unscale
    pred = output_scaler.inverse_transform(pred)
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
    if graph:
        plt.plot(bnh, label='Buy and Hold')
        plt.plot(strategy, label='Strategy')
        plt.legend()
        plt.show()
    return data["accum_buy_and_hold"], data["accum_strategy_returns"] # return ultimate profits
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:47:46 2022

@author: dylan
"""
import yfinance
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing
import rsi_calculator
from dateutil.relativedelta import relativedelta
from datetime import datetime
import math
import back_tester
import seaborn as sns

np.random.seed(314)
STATE = np.random.get_state()
DAYS_IN_FUTURE_TO_PREDICT = 1
HISTORY = '10y'
INTERVAL = '1d' # 1h: max 2y
FIRST_DATE = (datetime.today() - relativedelta(days=(15 * 253) - 7)).strftime('%Y-%m-%d')
TICKER = 'SPY'
FEATURES = ['close', 'wr', 'volume', '5d-rolling-sma', '20d-rolling-avg',
       'bollinger_up_20', 'bollinger_down_20', 'bollinger_up_10',
       'bollinger_down_10', 'bollinger_up_5', 'bollinger_down_5', 'rsi',
       'highs', 'lows']

def load_close_data(ticker, period):
    data = yfinance.download(ticker, period=period, interval=INTERVAL)
    close_data = data[['Close']]
    date_data = data.index.to_frame(index=False, name='Date')  # Extract Date from the index
    high_data = data[['High']]
    low_data = data[['Low']]
    volume = data[['Volume']]
    return close_data, date_data, high_data, low_data, volume
   
def get_train_and_test_data(d):
    train_len = math.ceil(d.shape[0] * 0.9)
    return d[:train_len, :], d[train_len:, :]

def split_closes_into_X_and_y(closes):
    X = []
    y = []
    for i in range(0, len(closes)):
        X.append(closes.iloc[i:i+1].values[-1])
        try:
            gl = -1
            if closes.iloc[i+DAYS_IN_FUTURE_TO_PREDICT].values[-1] > closes.iloc[i].values[-1]:
                gl = 1
            y.append(gl)
        except:
            y.append(None) # this None value will get removed later via X.dropna()
    X, y = np.asarray(X), np.asarray(y)
    return X, y

def get_bollinger_bands(X, rate=20):
    sma = (X.iloc[:, 0]).rolling(rate).mean()
    std = X.iloc[:, 0].rolling(rate).std()
    bollinger_up = sma + (std * 2) # Calculate top band
    bollinger_down = sma - (std * 2) # Calculate bottom band
    return sma, std, bollinger_up, bollinger_down

def calculate_ema(X, days, smoothing=2):
    ema = []
    for i in range(days-1):
        ema.append(None) # this will get removed later - it's just so that initial dimensions will match
    if (X.shape[0] <= 0): return ema # if there's no data, return an empty list
    ema.append(sum(X.iloc[:days, 0]) / X.shape[0])
    for close in X.iloc[days:, 0]:
        ema.append((close * (smoothing / (1 + days))) + (ema[-1] * (1 - (smoothing / (1 + days)))))
    return ema

def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max().values 
    lowl = low.rolling(lookback).min().values
    close = close.values.reshape(len(close), 1)
    #lowl = lowl.dropna().values
    #highh = highh.dropna().values
    #close = close.dropna().values.reshape(len(close), 1)
    highh = highh[:len(close)]
    lowl = lowl[:len(close)]
    wr = -100 * (np.subtract(highh, close) / np.subtract(highh, lowl))
    return wr

def prepare_data():
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(TICKER, HISTORY)
    X, y = split_closes_into_X_and_y(close_data)
    X = pd.DataFrame(X, columns=['close'])
    
    # get bollinger bands
    sma, std, up20, down20 = get_bollinger_bands(X, 20)
    _, _, up10, down10 = get_bollinger_bands(X, 10)
    sma_5, _, up5, down5 = get_bollinger_bands(X, 5)
    
    # get emas
    ema10 = calculate_ema(X, 10)
    ema20 = calculate_ema(X, 20)
    ema50 = calculate_ema(X, 50)
    
    # get Williams %R
    wr = get_wr(highs, lows, X.iloc[:, 0], 14)
    
    # add Williams %R
    X['wr'] = wr

    # add volume
    X['volume'] = volume.values
    
    # add emas
    X['10-day-ema'] = ema10
    X['20-day-ema'] = ema20
    X['50-day-ema'] = ema50
    
    # add 7 day rolling 
    X['5d-rolling-sma'] = sma_5
    
    # add 20 moving avg
    X['20d-rolling-avg'] = sma
    
    # add bollinger bands
    X['bollinger_up_20'] = up20
    X['bollinger_down_20'] = down20
    X['bollinger_up_10'] = up10
    X['bollinger_down_10'] = down10
    X['bollinger_up_5'] = up5
    X['bollinger_down_5'] = down5
    
    # add rsi
    rsi = rsi_calculator.calculateRSI(TICKER, HISTORY, INTERVAL)
    X['rsi'] = rsi[1:]
    
    # add highs and lows
    X['highs'] = highs.values[0:]
    X['lows'] = lows.values[0:]

    # add y values for scaling
    X['y'] = y

    # remove na
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    feature_names = X.columns
    
    print(X.head())
    print(X.tail())
    
    X = np.array(X)
    
    # split into train, test
    train, test = get_train_and_test_data(X)

    # resplit into X, y
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    run_random_forest(trainX, trainy, testX, testy, feature_names)

    return _, trainX, trainy, testX, testy

def calculate_indicators():
    close_col = 'close'
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(TICKER, HISTORY)
    data, y = split_closes_into_X_and_y(close_data)
    data = pd.DataFrame(data, columns=[close_col])

    data['Volume'] = volume.values

    data["SMA20"] = data[close_col].rolling(20).mean()
    data["SMA50"] = data[close_col].rolling(50).mean()
    
    # Squeeze Pro: Bollinger Bands vs. Keltner Channels
    data["stddev"] = data[close_col].rolling(20).std()
    data["upper_bb"] = data["SMA20"] + 2 * data["stddev"]
    data["lower_bb"] = data["SMA20"] - 2 * data["stddev"]
    data["upper_kc"] = data["SMA20"] + 1.5 * data["stddev"]
    data["lower_kc"] = data["SMA20"] - 1.5 * data["stddev"]
    data["squeeze_pro"] = ((data["lower_bb"] > data["lower_kc"]) & (data["upper_bb"] < data["upper_kc"])).astype(int)

    # Percentage Price Oscillator (PPO)
    data["ema12"] = data[close_col].ewm(span=12, adjust=False).mean()
    data["ema26"] = data[close_col].ewm(span=26, adjust=False).mean()
    data["ppo"] = ((data["ema12"] - data["ema26"]) / data["ema26"]) * 100

    # Thermo (relative position over a range)
    data["thermo"] = (data[close_col] - data["SMA20"]) / (data["upper_bb"] - data["lower_bb"])

    # Decay (exponential decay of price changes)
    decay_factor = 0.9
    data["decay"] = data[close_col].diff().ewm(alpha=1 - decay_factor, adjust=False).mean()

    # Archer On-Balance Volume (OBV)
    data["volume_change"] = data["Volume"] * np.sign(data[close_col].diff())
    data["archer_obv"] = data["volume_change"].cumsum()

    # Bollinger Bands
    data["bb_mid"] = data["SMA20"]
    data["bb_upper"] = data["upper_bb"]
    data["bb_lower"] = data["lower_bb"]

    # Squeeze Indicator
    data["squeeze"] = ((data["bb_upper"] - data["bb_lower"]) < (1.5 * data["stddev"])).astype(int)

    # Ichimoku Indicator
    data["conversion_line"] = (data[close_col].rolling(9).max() + data[close_col].rolling(9).min()) / 2
    data["base_line"] = (data[close_col].rolling(26).max() + data[close_col].rolling(26).min()) / 2
    data["leading_span_a"] = ((data["conversion_line"] + data["base_line"]) / 2).shift(26)
    data["leading_span_b"] = ((data[close_col].rolling(52).max() + data[close_col].rolling(52).min()) / 2).shift(26)
    data["lagging_span"] = data[close_col].shift(-26)
    
    # add y values
    data['y'] = y
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    return data

def run_random_forest(X_train, y_train, X_test, y_test, feature_names):
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    accuracy_before = rf.score(X_test, y_test)
    print(f'Accuracy before feature selection: {accuracy_before:.2f}')
    # Extract feature importances
    importances = rf.feature_importances_
    feature_names = feature_names[:-1] # remove the y column
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # Select top N features (example selecting top 10 features)
    top_features = feature_importance_df.iloc[:10, 0].index
    X_train_selected = X_train[:, top_features]
    X_test_selected = X_test[:, top_features]
    # Train the Random Forest model with selected features
    rf_selected = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_selected.fit(X_train_selected, y_train)

    # Evaluate the model
    accuracy_after = rf_selected.score(X_test_selected, y_test)
    print(f'Accuracy after feature selection: {accuracy_after:.2f}')

def run_entire_model():
    X = calculate_indicators()
    feature_names = X.columns
    
    X = np.array(X)
    
    # split into train, test
    train, test = get_train_and_test_data(X)

    # resplit into X, y
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    run_random_forest(trainX, trainy, testX, testy, feature_names)

run_entire_model()
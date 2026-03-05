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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # prevent GPU on Render server
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import math
from scipy.stats import pearsonr
import back_tester
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps

# amount of past data the model can see; 1h is almost nothing for price prediction
SHORT_TERM_HISTORY = 12    # use the previous 12 intervals (e.g. hours) instead of just one
HOURS_IN_FUTURE_TO_PREDICT = 1
HISTORY = '2y'
INTERVAL = '1h' # 1h: max 2y
TICKER = 'CMCSA'

def load_close_data(ticker, period):
    data = yfinance.download(ticker, period=period, interval=INTERVAL)
    close_data = data[['Close']]
    date_data = data.index.to_frame(index=False, name='Date')  # Extract Date from the index
    high_data = data[['High']]
    low_data = data[['Low']]
    volume = data[['Volume']]
    return close_data, date_data, high_data, low_data, volume
    
def create_model(trainX):
    # simple 1‑layer LSTM with dropout and a small dense head; you can
    # stack more layers or try bidirectional variants later

    timesteps = trainX.shape[1]
    n_features = trainX.shape[2]

    model = keras.Sequential([
        keras.layers.Input(shape=(timesteps, n_features)),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),   # return_sequences=True for attention
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='linear') # sigmoid for binary classification, linear for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    """model.compile(
        optimizer=keras.optimizers.Adam(
        learning_rate=0.00001,   # drop from 0.001 to 0.00001
        clipnorm=1.0             # clip gradients to prevent explosion
        ),
        loss='binary_crossentropy', 
        metrics=['accuracy']) # for classification"""
    print(model.summary())
    return model

def train_model(trainX, trainy, testX, testy, batch_size=128, epochs=4, plot=False):
    model = create_model(trainX)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("training", exist_ok=True)
    checkpoint_path = "training/cp-{epoch:04d}.weights.h5"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        trainX, trainy,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(testX, testy),
        callbacks=[checkpointer, earlystop],
        verbose=1
    )
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print(f'Training MSE: {train_mse}, Testing MSE: {test_mse}')
    if plot:
        plt.figure()
        plt.title('Loss, MSE')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        plt.pause(1)
    return model

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
    highh = highh[:len(close)]
    lowl = lowl[:len(close)]
    wr = -100 * (np.subtract(highh, close) / np.subtract(highh, lowl))
    return wr

def calculate_rsi(closes, window_len):
    window_len = 14
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

def get_technical_indicators(ticker=TICKER):
    close_col = 'close'
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(ticker, HISTORY)
    data, y = split_closes_into_X_and_y(close_data)
    data = pd.DataFrame(data, columns=[close_col])

    data['Volume'] = volume.values[1:]

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
    # data["lagging_span"] = data[close_col].shift(-26)

    # get and add Williams %R
    wr = get_wr(highs, lows, data.iloc[:, 0], 14)
    data['wr'] = wr
    
    # add rsi
    rsi = calculate_rsi(close_data, 14)
    data['rsi'] = rsi[SHORT_TERM_HISTORY+1:]

    # add VWAP
    # data['vwap'] = (((data[close_col] + highs.values[1:] + lows.values[1:]) / 3) * data['Volume']).cumsum() / data['Volume'].cumsum()

    # this will get dropped next because the last
    # training example has a y value of None -> preserve it for prediction
    most_recent_X_row = data.iloc[-1, :]
    
    # add y values
    data['y'] = y

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    return data, most_recent_X_row

def get_data(ticker=TICKER):
    close_col = 'close'
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(ticker, HISTORY)
    data = pd.DataFrame(close_data.values, columns=[close_col])

    data['Date'] = dates.values.ravel()
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
    # data["volume_change"] = data["Volume"] * np.sign(data[close_col].diff())
    #data["archer_obv"] = data["volume_change"].cumsum()

    # Bollinger Bands
    data["bb_mid"] = data["SMA20"]
    data["bb_upper"] = data["upper_bb"]
    data["bb_lower"] = data["lower_bb"]

    # Squeeze Indicator
    # data["squeeze"] = ((data["bb_upper"] - data["bb_lower"]) < (1.5 * data["stddev"])).astype(int)

    # Ichimoku Indicator
    data["conversion_line"] = (data[close_col].rolling(9).max() + data[close_col].rolling(9).min()) / 2
    data["base_line"] = (data[close_col].rolling(26).max() + data[close_col].rolling(26).min()) / 2
    data["leading_span_a"] = ((data["conversion_line"] + data["base_line"]) / 2).shift(26)
    data["leading_span_b"] = ((data[close_col].rolling(52).max() + data[close_col].rolling(52).min()) / 2).shift(26)
    # data["lagging_span"] = data[close_col].shift(-26)

    # get and add Williams %R
    wr = get_wr(highs, lows, data.iloc[:, 0], 14)
    data['wr'] = wr
    
    # add rsi
    rsi = calculate_rsi(close_data, 14)
    data['rsi'] = rsi

    # add VWAP
    # data['vwap'] = (((data[close_col] + highs.values[1:] + lows.values[1:]) / 3) * data['Volume']).cumsum() / data['Volume'].cumsum()

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

    # extract dates
    dates = data['Date']
    data = data.drop(columns=['Date']) # remove dates
    return data, dates

def split_X_y(data):
    X = []
    y = []
    test_closes = []

    pct_changes = data['close'].pct_change().values
    for i in range(len(data) - SHORT_TERM_HISTORY):
        X.append(data.iloc[i:(i+SHORT_TERM_HISTORY)])

        # Predict next close
        y.append(float(data["close"].values[i+SHORT_TERM_HISTORY])) # predict next close

        # Predict direction
        #y.append((data["close"].values[i+SHORT_TERM_HISTORY] > data["close"].values[i+SHORT_TERM_HISTORY-1]).astype(int)) # up or down classification

        # Predict return
        #y.append(float(pct_changes[i+SHORT_TERM_HISTORY])) # predict next close
        test_closes.append(float(data["close"].values[i+SHORT_TERM_HISTORY]))
    return np.array(X), np.array(y), np.array(test_closes)

def get_old_technical_data():
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(TICKER, HISTORY)
    X, y = split_closes_into_X_and_y(close_data)
    X = pd.DataFrame(X, columns=[f'close-{i}' for i in range(SHORT_TERM_HISTORY, 0, -1)])
    
    # get dates
    dates_array = []
    for i in range(SHORT_TERM_HISTORY, len(dates)):
        dates_array.append(dates.iloc[i].name)
    
    # get bollinger bands
    sma, std, up20, down20 = get_bollinger_bands(X, 20)
    _, _, up10, down10 = get_bollinger_bands(X, 10)
    sma_5, _, up5, down5 = get_bollinger_bands(X, 5)
    
    # get emas
    ema10 = calculate_ema(X, 10)
    ema20 = calculate_ema(X, 20)
    ema50 = calculate_ema(X, 50)
    
    # get Williams %R
    wr = get_wr(highs, lows, X.iloc[:, SHORT_TERM_HISTORY-1], min(SHORT_TERM_HISTORY, 14))
    
    # add Williams %R
    X['wr'] = wr

    # add volume
    X['volume'] = volume.values[SHORT_TERM_HISTORY:]
    
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
    X['rsi'] = rsi[SHORT_TERM_HISTORY+1:]
    
    # add highs and lows
    X['highs'] = highs.values[SHORT_TERM_HISTORY:]
    X['lows'] = lows.values[SHORT_TERM_HISTORY:]

    #X = X.iloc[:, SHORT_TERM_HISTORY-1:]
    #check_redundant_features(X, y)

    # if we want to select specific features
    if type(feature_indices) != type(None):
        #feature_indices = [i + SHORT_TERM_HISTORY - 1 for i in feature_indices]
        #feature_indices = [i for i in range(SHORT_TERM_HISTORY)] + feature_indices
        feature_indices = [SHORT_TERM_HISTORY-1] + (list(feature_indices))
        X = X.iloc[:, feature_indices]

    # add y values for scaling
    X['y'] = y

    # remove na
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    return X

def prepare_data(ticker=TICKER):
    data, dates = get_data(ticker)
    X, y, test_closes = split_X_y(data)
    dates = dates[SHORT_TERM_HISTORY:]
        
    # split into train, test datasets
    split = int(0.9 * len(X))
    trainX = X[:split]
    trainy = y[:split]
    testX = X[split:] 
    testy = y[split:] 

    latest_X_row = data.iloc[-SHORT_TERM_HISTORY:]

    n_train = trainX.shape[0]
    n_test = testX.shape[0]
    timesteps = trainX.shape[1]
    n_features = trainX.shape[2]

    trainX_2d = trainX.reshape(-1, trainX.shape[2])
    testX_2d = testX.reshape(-1, testX.shape[2])
    latest_X_row_2d = latest_X_row.values

    print("Per-feature max:")
    for i, col in enumerate(data.columns):
        print(f"  {col}: max={trainX_2d[:, i].max():.2f}, min={trainX_2d[:, i].min():.2f}")

    # scale
    #input_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    input_scaler = preprocessing.RobustScaler()
    input_scaler.fit(trainX_2d)

    trainX = input_scaler.transform(trainX_2d).reshape(n_train, timesteps, n_features)
    trainX = np.clip(trainX, -3, 3)
    testX = input_scaler.transform(testX_2d).reshape(n_test, timesteps, n_features)
    testX = np.clip(testX, -3, 3)
    latest_X_row = input_scaler.transform(latest_X_row_2d).reshape(1, timesteps, n_features)
    
    output_scaler = preprocessing.RobustScaler()
    output_scaler.fit(trainy.reshape(-1, 1))
    trainy = output_scaler.transform(trainy.reshape(-1, 1)).reshape(-1, 1)
    testy = output_scaler.transform(testy.reshape(-1, 1)).reshape(-1, 1)

    trainy = trainy.reshape(-1, 1)
    testy = testy.reshape(-1, 1)

    print(f'trainX: {trainX.shape}, testX: {testX.shape}, trainy: {trainy.shape}, testy: {testy.shape}')
    return input_scaler, output_scaler, trainX, trainy, testX, testy, latest_X_row, dates[split:], test_closes[split:]

def load_the_model(ticker=TICKER):
    _, output_scaler, trainX, _, _, _, _ = prepare_data()
    model = create_model(trainX)
    model.load_weights('./training/cp-0005.weights.h5')
    return model, output_scaler

def graph_prediction(y, predictions):
    y = pd.DataFrame(y, columns=['Close'])
    y['Predictions'] = predictions
    plt.figure(figsize=(8,4))
    plt.title(f'Model for {TICKER}')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(y[['Close', 'Predictions']])
    plt.legend(['Val', 'Predictions'], loc='lower right')
    plt.show()
       
def test_statistical_significance_of_features():
    _, _, trainX, trainy, _, _, _ = prepare_data()
    
    for i in range(0, trainX.shape[1]):
        X_var = trainX['dow-jones']
        y_var = trainy
        correlation, p = pearsonr(X_var, y_var)
        print(i, '===> Correlation coefficient: ', correlation, 'p-value: ', p)

def backtest_strategy(df, predictions):

    class PredictionStrategy(Strategy):
        buy_threshold = 0.01
        sell_threshold = 0.01

        def init(self):
            self.signals = self.I(lambda: predictions, name='LSTM Predictions')

        def next(self):
            prediction = self.signals[-1]
            current = self.data.Close[-1]
            
            if prediction > current * (1 + self.buy_threshold):
                if not self.position.is_long:
                    self.buy()
            elif prediction < current * (1 - self.sell_threshold):
                if not self.position.is_short:
                    self.sell()
            else:
                self.position.close()
    
    class DirectionStrategy(Strategy):

        def init(self):
            self.signals = self.I(lambda: predictions, name='LSTM Predictions')

        def next(self):
            prediction = self.signals[-1]
            
            if prediction > 0:
                if not self.position.is_long:
                    self.buy()
            elif prediction <= 0:
                if not self.position.is_short:
                    self.sell()

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
    buy_thresholds = [float(round(i, 2)) for i in np.arange(0.15, 0.9, 0.02)]
    sell_thresholds = [float(round(i, 2)) for i in np.arange(0.15, 0.9, 0.02)]
    stats, heatmap = train_bt.optimize(
        buy_threshold=buy_thresholds,
        sell_threshold=sell_thresholds, 
        constraint=lambda buy_threshold, sell_threshold: buy_threshold < sell_threshold, # buy threshold must be less than sell threshold
        maximize='Equity Final [$]',
        max_tries=None,
        random_state=42,
        return_heatmap=True,
        return_optimization=False
    )
    plot_heatmaps(heatmap, agg='mean')
    optimal_buy_threshold = stats.at['_strategy'].buy_threshold
    optimal_sell_threshold = stats.at['_strategy'].sell_threshold
    print(f'Best buy threshold: {optimal_buy_threshold}, Best sell threshold: {optimal_sell_threshold}')
    
    RSIStrategy.buy_threshold = optimal_buy_threshold
    RSIStrategy.sell_threshold = optimal_sell_threshold
    RSIStrategy.pred_data = predictions[split:].flatten()
    test_bt = Backtest(test_df, RSIStrategy, cash=10000, commission=0, finalize_trades=True)
    stats = test_bt.run()
    print(stats)
    test_bt.plot()


def load_data_train_and_predict(ticker=TICKER, graph=True):
    input_scaler, output_scaler, trainX, trainy, testX, testy, latest_X_row, test_dates, test_closes = prepare_data(ticker)
    print(f'Training samples: {str(len(trainy))}, testing samples: {str(len(testy))}')

    # 3. Check feature scale
    print("X mean:", trainX.mean())
    print("X std: ", trainX.std())
    print("X min: ", trainX.min())
    print("X max: ", trainX.max())

    model = train_model(trainX, trainy, testX, testy, graph)

    test_predictions = model.predict(testX)
    print(test_predictions.shape)
    latest_future_prediction = model.predict(latest_X_row)

    # unscale
    latest_future_prediction = output_scaler.inverse_transform(latest_future_prediction.reshape(-1, 1))
    test_predictions = output_scaler.inverse_transform(test_predictions.reshape(-1, 1))
    testy = output_scaler.inverse_transform(testy.reshape(-1, 1))
    print(f'prediction tail: {test_predictions[-5:, :]}, test tail: {testy[-5:, :]}')
    if graph:
        graph_prediction(testy, test_predictions)
    
    se = np.sqrt(np.mean(((test_predictions - testy)**2)))
    
    print(f'Prediction 1 hour from now: {str(latest_future_prediction)}')
    print(f'Error: {se}')

    #bnh_returns, strategy_returns = back_tester.test_strategy(test_predictions, testy, graph=graph)
    #print(f'Buy and hold returns: {bnh_returns.iloc[-1]}, Strategy returns: {strategy_returns.iloc[-1]}')

    # backtest
    df = pd.DataFrame({
        'Open': test_closes.flatten(),
        'High': test_closes.flatten(),
        'Low': test_closes.flatten(),
        'Close': test_closes.flatten(),
        'Volume': np.ones_like(test_closes.flatten())
    }, index=pd.DatetimeIndex(test_dates.values.flatten()))
    backtest_strategy(df, test_predictions.flatten())

    return latest_future_prediction[-1] # return the prediction for the next future time interval


def test_rsi_strategy(ticker=TICKER, graph=True):
    close_col = 'Close'
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(ticker, HISTORY)
    data = pd.DataFrame(close_data.values, columns=[close_col])

    data['Date'] = dates.values.ravel()
    data['Open'] = close_data.values
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
        'Open': close_data.values.flatten(),
        'High': data["High"].values.flatten(),
        'Low': data["Low"].values.flatten(),
        'Close': close_data.values.flatten(),
        'Volume': data["Volume"].values.flatten()
    }, index=pd.DatetimeIndex(dates.values.flatten()))
    backtest_strategy(df, data['rsi'].values.flatten())


def main():
    #load_data_train_and_predict()
    test_rsi_strategy()
    pass

if __name__=='__main__':
    main()

from imports import *
from data_model import *
from ml_model import *
import warnings
class PortfolioAnalyzer():
    def __init__(self, data_model=None, ml_model=None,ml_model_large=None):
        self.data_model = data_model
        self.ml_model = ml_model
        self.ml_model_large = ml_model_large
        if self.ml_model:
            try:
                self.ml_model_large = self.ml_model.models['model_large']
            except:
                warnings.warn("ml_model_large not found in ml_model.models!!")
    def run_portfolio(self, threshold = 0.03, y_hat_df=None, ytest=None):
        if y_hat_df is None:
            y_hat_df = self.y_hat_df
        if ytest is None:
            ytest = self.data_model.ytest[-1].iloc[1:]
        ywindow = 1
        l2 = [*y_hat_df.rolling(window=ywindow + 1)]
        results = []

        for i, df6 in enumerate(l2[ywindow:]):
            if i % 1000 == 0:
                print(i)
            if df6[0].values[-1] > (1. + threshold) * df6[0].values[0]:
                k = 1
            elif df6[0].values[0] > (1. + threshold) * df6[0].values[-1]:
                k = -1
            else:
                k = 0

            results.append(k)
        pf_kwargs = dict(
            close=ytest[-1].iloc[1:],
            size=results,
            fees=0.0003,
            init_cash=10000.
        )
        self.pf = vbt.Portfolio.from_orders(**pf_kwargs)
        print(f"PnL: {self.pf.entry_trades.pnl.sum()}")
    def predict(self, threshold = 0.03, large=True, xtest=None, close=None, ytest=None):
        if xtest is None:
            xtest = self.data_model.xtest.values
            xtest = xtest.reshape(xtest.shape[0] // 60, 60, xtest.shape[1]).astype('float32')
        if ytest is None:
            ytest = self.data_model.ytest
        if large:
            #xtest = [xtest, xtest, xtest, xtest, xtest]
            xtest = [xtest, xtest, xtest]
        y_hat = self.ml_model_large.predict(xtest)
        self.y_hat_df = pd.DataFrame(y_hat, index =ytest.index)
        self.run_portfolio(threshold=threshold, y_hat_df=self.y_hat_df, ytest=ytest)
    def predict_2(self, xtest=None, large=True, ytest=None, close=None):
        if xtest is None:
            xtest = self.data_model.xtest.values
            xtest = xtest.reshape(xtest.shape[0] // 60, 60, xtest.shape[1]).astype('float32')
        if large:
            xtest = [xtest, xtest, xtest]
        ytest = self.data_model.ytest if ytest is None else ytest
        if close is None:
            self.close = self.data_model.df.loc[ytest.index, 'close']
        y_hat = self.ml_model_large.predict(xtest)
        self.trade_df = pd.DataFrame(y_hat, index=ytest.index, columns="trade")
    def run_portfolio_2(self, close, trades, pf_args={}):
        portfolio_args = dict(
            close=close,
            size=trades,
            fees=0.0003,
            init_cash=10000.
        )
        self.pf = vbt.Portfolio.from_orders(**portfolio_args)
        print(f"PNL: {self.pf.entry_trades.pnl.sum()}")


    def plot(self):
        entry_trades = self.pf.entry_trades.records_readable.groupby('Entry Timestamp').sum()
        exit_trades = self.pf.entry_trades.records_readable.groupby('Exit Timestamp').sum()
        plt.plot(self.data_model.ytest[-1])


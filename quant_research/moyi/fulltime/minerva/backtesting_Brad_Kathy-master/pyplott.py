import datetime
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web



class Visualization(object):
    def __init__(self, bt_csv_dir, start_date_str):
        self.equity_curve = pd.read_csv(bt_csv_dir + "equity.csv")
        self.equity_curve.drop_duplicates("datetime", inplace=True)
        self.equity_curve = self.equity_curve[self.equity_curve['datetime'] > start_date_str]
        self.equity_curve.index = self.equity_curve['datetime']
        self.equity_curve = self.equity_curve[self.equity_curve['total'].map(lambda x: str(x) != "nan")]
        self.equity_curve.drop(['datetime'], axis=1, inplace=True)
        self.equity_curve.index = pd.to_datetime(self.equity_curve.index, format='%Y-%m-%d')
        self.plot_size = [18, 6]

    def plot_commissions(self):
        self.equity_curve['commission'].plot(figsize=self.plot_size)
        plt.show()

    def plot_cash(self):
        self.equity_curve['cash'].plot(figsize=self.plot_size)
        plt.show()

    def plot_total(self):
        self.equity_curve['total'].plot(figsize=self.plot_size)
        plt.show()

    def plot_returns(self):
        self.equity_curve['returns'].plot(figsize=self.plot_size)
        plt.show()

    def plot_equity_curve(self):
        self.equity_curve['equity_curve'].plot(figsize=self.plot_size)
        plt.show()

    def plot_drawdown(self):
        self.equity_curve['drawdown'].plot(figsize=self.plot_size)
        plt.show()

    def return_equity_df(self):
        return self.equity_curve

    def get_SPX(self):
        start_dt = self.equity_curve.index.min()
        end_dt = self.equity_curve.index.max() - datetime.timedelta(1)
        sp500 = web.DataReader('^GSPC', 'yahoo', start_dt, end_dt)['Adj Close']
        return sp500

    def plot_comp_SPX(self, sp500):
        comb_df = self.equity_curve[['returns']]
        comb_df['ret_sp500'] = sp500.pct_change()
        cumprod_df = (comb_df.dropna() + 1).cumprod()
        cumprod_df.plot(figsize=self.plot_size)
        plt.show()


equity_folder = "./"
start_date = datetime.datetime(2010, 4, 1, 0, 0, 0)
start_date_str = str(start_date)
visualization = Visualization(equity_folder, start_date_str)

visualization.plot_equity_curve()

sp500 = visualization.get_SPX()
visualization.plot_comp_SPX(sp500)
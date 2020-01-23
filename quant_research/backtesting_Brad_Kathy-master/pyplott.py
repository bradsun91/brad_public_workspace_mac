import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio import *

obj = Portfolio()











class Visualization(Portfolio):

    def __init__ (self):
        super().__init__()
    #     inherit the self.equity_curve from portfolio

    def tsPlot(self, column_name):
        self.equity_curve[column_name].plot()
        plt.show()


if __name__ == "__main__":
    # port = Visualization.create_equity_curve_dataframe
    Visualization.tsPlot('equity_curve')
    # obj.tsPlot('equity_curve')



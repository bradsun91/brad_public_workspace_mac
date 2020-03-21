import numpy as np
import matplotlib.pyplot as plt

class Payoff(object):
    def __init__(self, S0, K, premium, is_put=False):
        self.S0 = S0
        self.K = K
        self.premium = premium
        self.is_put = is_put 

    def __repr__(self):
        return 'Option(S0: {}, K: {}, premium: {}, is_put: {})'.format(self.S0, self.K, self.premium, self.is_put)

    def __str__(self):
        return 'Option(S0: {}, K: {}, premium: {}, is_put: {})'.format(self.S0, self.K, self.premium, self.is_put)
    
    def calculate_payoff(self, position=None, sT=None):
        """ 
        sT is a range of possible stock prices
        to help us visualize option payoffs
        """
        # return a set of payoffs given a range of S0 prices
        self.position = position 
        
        if isinstance(sT,np.ndarray):
            if not self.is_put:
                payoff = np.where(sT > self.K, sT - self.K, 0) - self.premium
                self.payoff = payoff
                if self.position == 'short':
                    self.payoff = -payoff
                return self.payoff
            else:
                payoff = np.where(sT < self.K,  self.K - sT, 0) - self.premium
                self.payoff = payoff
                if self.position == 'short':
                    self.payoff = -payoff
                return self.payoff

        # return current payoffs given the current S0 price 
        else:
            if not self.is_put:
                payoff = max(self.S0 - self.K, 0) - self.premium
                self.payoff = payoff 
                if position == 'short':
                    self.payoff = -payoff
                return self.payoff
            else:
                payoff = max(self.K - self.S0, 0) - self.premium
                self.payoff = payoff 
                if position == 'short':
                    self.payoff = -payoff
                return self.payoff
    
    def current_payoff(self):
        if not self.is_put:
            payoff = max(self.S0 - self.K, 0) - self.premium
            if self.position == 'short':
                payoff = -payoff
            return payoff
        else:
            payoff = max(self.K - self.S0, 0) - self.premium
            if self.position == 'short':
                payoff = -payoff
            return payoff
            
    def visualize(self, sT):
        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.axvline(self.S0, label='spot',color='r')
        ax.plot(sT,self.payoff,label='{} {} {}'.format(self.position, self.K, 'put' if self.is_put else 'call'))

        plt.title('Naked {}'.format('put' if self.is_put else 'call'))
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and Loss')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    stock1 = Payoff(100, 100, 0.5, False)
    stock2 = Payoff(100, 110, 0.3, True)
    stock3 = Payoff(100, 90, 0.8, False)
    print(stock1)
    print(stock1.calculate_payoff())
    print(stock2)
    print(stock2.calculate_payoff())
    print(stock3)
    print(stock3.calculate_payoff())

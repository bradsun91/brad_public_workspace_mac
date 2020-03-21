import numpy as np
import math

class Option(object):
    def __init__(self, S0, K, r=0.05, T=1, N=2, pu=0, pd=0, div=0, sigma=0, is_put=False, is_am=False):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1,N)
        self.STs = [] # Declare the stock price tree
        self.is_am = is_am
        self.is_put = is_put

        # optial parameters
        self.pu, self.pd = pu, pd
        self.div = div
        self.sigma = sigma

    @property
    def dt(self):
        """single time step, in years"""
        return self.T/float(self.N)

    @property 
    def df(self):
        """the discount factor"""
        return math.exp(-(self.r-self.div)*self.dt)

class Binomial(Option):
    def setup_parameters(self):
        self.M = self.N + 1
        self.u = 1 + self.pu
        self.d = 1 - self.pd
        self.qu = (math.exp((self.r - self.div)* self.dt) - self.d)/ (self.u - self.d)
        self.qd = 1 - self.qu 

    def init_stock_price_tree(self):
        # Initialize a 2D tree at T=0
        self.STs = [np.array([self.S0])]

        # simulate the possible stock prices path
        for _ in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                (prev_branches*self.u,
                    [prev_branches[-1]*self.d]))
            self.STs.append(st) # add nodes at each time step

    def init_payoffs_tree(self):
        if not self.is_put:
            return np.maximum(0, self.STs[self.N]-self.K)
        else:
            return np.maximum(0, self.K-self.STs[self.N])
    
    def check_early_exercise(self, payoffs, node):
        if not self.is_put:
            return np.maximum(payoffs, self.STs[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])
    
    def traverse_tree(self, payoffs):
        for i in reversed(range(self.N)):
            # not exercising options
            payoffs = (payoffs[:-1] * self.qu +
                        payoffs[1:]*self.qd) * self.df

            # payoffs from exercising American options
            if self.is_am:
                payoffs = self.check_early_exercise(payoffs, i)
        
        return payoffs

    def begin_tree_traversal (self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        return payoffs[0]

class CRR(Binomial):
    def setup_parameters(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r - self.div) * self.dt) -
                    self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

if __name__ == '__main__':
    long_put = Binomial(50, 52, r =0.05, T=2, N=2, pu=0.2, pd=0.2, is_put=True, is_am=True).price()
    eu_put = CRR(50, 52, r =0.05, T=2, N=2, sigma=0.3, is_put=True).price()
    print(long_put)
    print(eu_put)

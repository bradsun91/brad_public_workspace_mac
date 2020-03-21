import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from payoff import Payoff
from Binomial_CRR import CRR

class Strategy(object):
    def __init__(self, S0, r, T, N, div, sT, model, is_am, transaction_cost=0.65):
        self.S0 = S0
        self.r = r
        self.N = N
        self.T = T
        self.div = div
        self.sT = sT
        self.model = model 
        self.is_am = is_am
        self.transaction_cost = transaction_cost

    def _option(self, K, sigma, is_put, position=None):
        pass

    def delta(self, K, sigma, is_put, num_contract=1, position='long'):
        option_forward = self.model(self.S0+1, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        option_backward = self.model(self.S0-1, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price() 
        delta = (option_forward-option_backward)/2
        if position == 'long':
            return float(delta) * 100 * num_contract
        else: 
            return -1. * float(delta) * 100 * num_contract

    def gamma(self, K, sigma, is_put, num_contract=1, position='long'):
        option = self.model(self.S0, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        # option_forward = self.model(self.S0+1, K, r=self.r, T=self.T, N=self.N, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        # option_backward = self.model(self.S0-1, K, r=self.r, T=self.T, N=self.N, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        option_forward = self.model(self.S0+2, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        option_backward = self.model(self.S0-2, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price() 
        delta_foward = (option_forward-option)/2
        delta_backward = (option-option_backward)/2
        gamma = (delta_foward-delta_backward)/2
        
        if position == 'long':
            return float(gamma) * 100 * num_contract
        else: 
            return -1. * float(gamma) * 100 * num_contract
    
    def theta(self, K, sigma, is_put, num_contract=1, position='long'):
        option_forward = self.model(self.S0, K, r=self.r, T=self.T-1/365, N=self.N-1, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        option_backward = self.model(self.S0, K, r=self.r, T=self.T+1/365, N=self.N+1, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price() 
        theta = (option_forward-option_backward)/2

        if position == 'long':
            return float(theta) * 100 * num_contract
        else:
            return -1. * float(theta) * 100 * num_contract

    def vega(self, K, sigma, is_put, num_contract=1, position='long'):
        option_forward = self.model(self.S0, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma+0.01, is_put=is_put, is_am=self.is_am).price()
        option_backward = self.model(self.S0, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma-0.01, is_put=is_put, is_am=self.is_am).price() 
        theta = (option_forward-option_backward)/2
       
        if position == 'long':
            return float(theta) * 100 * num_contract
        else: 
            return -1. * float(theta) * 100 * num_contract

    def rho(self, K, sigma, is_put, num_contract=1, position='long'):
        option_forward = self.model(self.S0, K, r=self.r+0.01, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price()
        option_backward = self.model(self.S0, K, r=self.r-0.01, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=is_put, is_am=self.is_am).price() 
        theta = (option_forward-option_backward)/2
        if position == 'long':
            return float(theta) * 100 * num_contract
        else:
            return -1. * float(theta) * 100 * num_contract

    def naked_call(self, K, sigma, position, num_contract=1):
        premium = self.model(self.S0, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=False, is_am=self.is_am).price()
        
        call = Payoff(self.S0, K, premium, is_put=False)
        payoff = ((call.calculate_payoff(position, self.sT) * 100) - self.transaction_cost) * num_contract
        current_payoff = (call.current_payoff() * 100 - self.transaction_cost) * num_contract

        print('# {} {} Call:'.format(position.title(), K), call)
        print('-- Delta:', round(self.delta(K, sigma, False, num_contract, position), 4))
        print('-- Gamma:', round(self.gamma(K, sigma, False, num_contract, position), 4))
        print('-- Theta:', round(self.theta(K, sigma, False, num_contract, position), 4))
        print('-- Vega:', round(self.vega(K, sigma, False, num_contract, position), 4))
        print('-- Rho:', round(self.rho(K, sigma, False, num_contract, position), 4))
        print('-- Profit and Loss:', round(current_payoff, 4))
        print('\n')

        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.axvline(self.S0, label='spot',color='r')
        ax.plot(self.sT, payoff,label='{} {} call'.format(position, K))

        plt.title('Naked Call')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and Loss')
        plt.legend()
        plt.grid()
        plt.show()

        return current_payoff

    def naked_put(self, K, sigma, position, num_contract=1):
        premium = self.model(self.S0, K, r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigma, is_put=True, is_am=self.is_am).price()
        
        put = Payoff(self.S0, K, premium, is_put=True)
        payoff = (put.calculate_payoff(position, self.sT) * 100 - self.transaction_cost) * num_contract
        current_payoff = (put.current_payoff() * 100 - self.transaction_cost) * num_contract

        print('# {} {} Put:'.format(position.title(), K), put)
        print('-- Delta:', round(self.delta(K, sigma, False, num_contract, position), 4))
        print('-- Gamma:', round(self.gamma(K, sigma, False, num_contract, position), 4))
        print('-- Theta:', round(self.theta(K, sigma, False, num_contract, position), 4))
        print('-- Vega:', round(self.vega(K, sigma, False, num_contract, position), 4))
        print('-- Rho:', round(self.rho(K, sigma, False, num_contract, position), 4))
        print('-- Profit and Loss:', round(current_payoff, 4))
        print('\n')
        
        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.axvline(self.S0, label='spot',color='r')
        ax.plot(self.sT, payoff,label='{} {} put'.format(position, K))

        plt.title('Naked Call')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and Loss')
        plt.legend()
        plt.grid()
        plt.show()

        return current_payoff
        
    def iron_condor(self, strikes, sigmas, num_contract=1):
        #long call
        premium_long_call = self.model(self.S0, strikes[0], r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigmas[0], is_put=False, is_am=self.is_am).price()
        long_call = Payoff(self.S0, strikes[0], premium_long_call, is_put=False)
        long_call_payoff = (long_call.calculate_payoff('long', self.sT) * 100 - self.transaction_cost) * num_contract 

        #short call
        premium_short_call = self.model(self.S0, strikes[1], r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigmas[1], is_put=False, is_am=self.is_am).price()
        short_call = Payoff(self.S0, strikes[1], premium_short_call, is_put=False)
        short_call_payoff = (short_call.calculate_payoff('short', self.sT) * 100 - self.transaction_cost) * num_contract

        #long put 
        premium_long_put = self.model(self.S0, strikes[2], r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigmas[2], is_put=True, is_am=self.is_am).price()
        long_put = Payoff(self.S0, strikes[2], premium_long_put, is_put=True)
        long_put_payoff = (long_put.calculate_payoff('long',self.sT) * 100 - self.transaction_cost) * num_contract

        #short put 
        premium_short_put = self.model(self.S0, strikes[3], r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigmas[3], is_put=True, is_am=self.is_am).price()
        short_put = Payoff(self.S0, strikes[3], premium_short_put, is_put=True)
        short_put_payoff = (short_put.calculate_payoff('short',self.sT) * 100 - self.transaction_cost) * num_contract

        #Iron Condor 
        payoff = long_call_payoff + short_call_payoff + long_put_payoff + short_put_payoff
        current_payoff = ((long_call.current_payoff() + short_call.current_payoff() + long_put.current_payoff() + short_put.current_payoff()) * 100 - 4 * self.transaction_cost) * num_contract

        print('# Iron Condor Strategy:')
        print('-- Long {} Call:'.format(strikes[0]), long_call)
        print('-- Short {} Call:'.format(strikes[1]), short_call)
        print('-- Long {} Put:'.format(strikes[2]), long_put)
        print('-- Short {} Put:'.format(strikes[3]), short_put)
        print('-- Delta:', round(self.delta(strikes[0], sigmas[0], False, num_contract) + self.delta(strikes[1], sigmas[1], False, num_contract, 'short') + self.delta(strikes[2], sigmas[2], True, num_contract) + self.delta(strikes[3], sigmas[3], True, num_contract, 'short'), 4))
        print('-- Gamma:', round(self.gamma(strikes[0], sigmas[0], False, num_contract) + self.gamma(strikes[1], sigmas[1], False, num_contract, 'short') + self.gamma(strikes[2], sigmas[2], True, num_contract) + self.gamma(strikes[3], sigmas[3], True, num_contract, 'short'), 4))
        print('-- Theta:', round(self.theta(strikes[0], sigmas[0], False, num_contract) + self.theta(strikes[1], sigmas[1], False, num_contract, 'short') + self.theta(strikes[2], sigmas[2], True, num_contract) + self.theta(strikes[3], sigmas[3], True, num_contract, 'short'), 4))
        print('-- Vega:', round(self.vega(strikes[0], sigmas[0], False, num_contract) + self.vega(strikes[1], sigmas[1], False, num_contract, 'short') + self.vega(strikes[2], sigmas[2], True, num_contract) + self.vega(strikes[3], sigmas[3], True, num_contract, 'short'), 4))
        print('-- Rho:', round(self.rho(strikes[0], sigmas[0], False, num_contract) + self.rho(strikes[1], sigmas[1], False, num_contract, 'short') + self.rho(strikes[2], sigmas[2], True, num_contract) + self.rho(strikes[3], sigmas[3], True, num_contract, 'short'), 4))
        print('-- Profit and Loss:', round(current_payoff, 4))
        print('\n')

        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.plot(self.sT,long_call_payoff,'--',label='Long Call',color='g')
        ax.plot(self.sT,short_call_payoff,'--',label='Short Call',color='r')
        ax.plot(self.sT,long_put_payoff,'--',label='Long Put',color='y')
        ax.plot(self.sT,short_put_payoff,'--',label='Short Put',color='m')
        ax.plot(self.sT,payoff,label='Iron Condor')

        plt.title('Iron Condor Strategy')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and loss')
        plt.legend()
        plt.grid()

        plt.show()
        return current_payoff

    def call_spread(self, strikes, sigmas, num_contract=1):
        #long call
        premium_long_call = self.model(self.S0, strikes[0], r=self.r, T=self.T, N=self.N, sigma=sigmas[0], is_put=False, is_am=self.is_am).price()
        long_call = Payoff(self.S0, strikes[0], premium_long_call, is_put=False)
        long_call_payoff = (long_call.calculate_payoff('long', self.sT) * 100 - self.transaction_cost) * num_contract

        #short call
        premium_short_call = self.model(self.S0, strikes[1], r=self.r, T=self.T, N=self.N, sigma=sigmas[1], is_put=False, is_am=self.is_am).price()
        short_call = Payoff(self.S0, strikes[1], premium_short_call, is_put=False)
        short_call_payoff = (short_call.calculate_payoff('short', self.sT) * 100 - self.transaction_cost) * num_contract

        #call spread 
        payoff = long_call_payoff + short_call_payoff
        current_payoff = ((long_call.current_payoff() + short_call.current_payoff()) * 100 - 2 * self.transaction_cost) * num_contract

        print('# {} Call Spread Strategy:'.format('Bull' if strikes[0] < strikes[1] else 'Bear'))
        print('-- Long {} Call:'.format(strikes[0]), long_call)
        print('-- Short {} Call:'.format(strikes[1]), short_call)
        print('-- Delta:', round(self.delta(strikes[0], sigmas[0], False, num_contract) + self.delta(strikes[1], sigmas[1], False, num_contract, 'short'), 4))
        print('-- Gamma:', round(self.gamma(strikes[0], sigmas[0], False, num_contract) + self.gamma(strikes[1], sigmas[1], False, num_contract, 'short'), 4))
        print('-- Theta:', round(self.theta(strikes[0], sigmas[0], False, num_contract) + self.theta(strikes[1], sigmas[1], False, num_contract, 'short'), 4))
        print('-- Vega:', round(self.vega(strikes[0], sigmas[0], False, num_contract) + self.vega(strikes[1], sigmas[1], False, num_contract, 'short'), 4))
        print('-- Rho:', round(self.rho(strikes[0], sigmas[0], False, num_contract) + self.rho(strikes[1], sigmas[1], False, num_contract, 'short'), 4))
        print('-- Profit and Loss:', round(current_payoff, 4))
        print('\n')

        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.plot(self.sT,long_call_payoff,'--',label='Long Call',color='g')
        ax.plot(self.sT,short_call_payoff,'--',label='Short Call',color='r')
        ax.plot(self.sT,payoff,label='{} Call Spread'.format('Bull' if strikes[0] < strikes[1] else 'Bear'))

        plt.title('{} Call Spread Strategy'.format('Bull' if strikes[0] < strikes[1] else 'Bear'))
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and loss')
        plt.legend()
        plt.grid()

        plt.show()
        return current_payoff

    def put_spread(self, strikes, sigmas, num_contract=1):
        #long put 
        premium_long_put = self.model(self.S0, strikes[0], r=self.r, T=self.T, N=self.N, sigma=sigmas[0], is_put=True, is_am=self.is_am).price()
        long_put = Payoff(self.S0, strikes[0], premium_long_put, is_put=True)
        long_put_payoff = (long_put.calculate_payoff('long',self.sT) * 100 - self.transaction_cost) * num_contract

        #short put 
        premium_short_put = self.model(self.S0, strikes[1], r=self.r, T=self.T, N=self.N, sigma=sigmas[1], is_put=True, is_am=self.is_am).price()
        short_put = Payoff(self.S0, strikes[1], premium_short_put, is_put=True)
        short_put_payoff = (short_put.calculate_payoff('short',self.sT) * 100 - self.transaction_cost) * num_contract

        #put spread 
        payoff = long_put_payoff + short_put_payoff
        current_payoff = ((long_put.current_payoff() + short_put.current_payoff()) * 100 - 2 * self.transaction_cost) * num_contract

        print('# {} Put Spread Strategy'.format('Bull' if strikes[0] < strikes[1] else 'Bear'))
        print('-- Long {} Put:'.format(strikes[0]), long_put)
        print('-- Short {} Put:'.format(strikes[1]), short_put)
        print('-- Delta:', round(self.delta(strikes[0], sigmas[0], True, num_contract) + self.delta(strikes[1], sigmas[1], True, num_contract, 'short'), 4))
        print('-- Gamma:', round(self.gamma(strikes[0], sigmas[0], True, num_contract) + self.gamma(strikes[1], sigmas[1], True, num_contract, 'short'), 4))
        print('-- Theta:', round(self.theta(strikes[0], sigmas[0], True, num_contract) + self.theta(strikes[1], sigmas[1], True, num_contract, 'short'), 4))
        print('-- Vega:', round(self.vega(strikes[0], sigmas[0], True, num_contract) + self.vega(strikes[1], sigmas[1], True, num_contract, 'short'), 4))
        print('-- Rho:', round(self.rho(strikes[0], sigmas[0], True, num_contract) + self.rho(strikes[1], sigmas[1], True, num_contract, 'short'), 4))
        print('-- Profit and Loss', round(current_payoff, 4))
        print('\n')

        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.plot(self.sT,long_put_payoff,'--',label='Long Put',color='g')
        ax.plot(self.sT,short_put_payoff,'--',label='Short Put',color='r')
        ax.plot(self.sT,payoff,label='{} Put Spread'.format('Bull' if strikes[0] < strikes[1] else 'Bear'))

        plt.title('{} Put Spread Strategy'.format('Bull' if strikes[0] < strikes[1] else 'Bear'))
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and loss')
        plt.legend()
        plt.grid()

        plt.show()
        return current_payoff

    def long_straddle(self, strikes, sigmas, num_contract=1):
        #long call
        premium_long_call = self.model(self.S0, strikes[0], r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigmas[0], is_put=False, is_am=True).price()
        long_call = Payoff(self.S0, strikes[0], premium_long_call, is_put=False)
        long_call_payoff = (long_call.calculate_payoff('long', self.sT) * 100 - self.transaction_cost) * num_contract

        #long put 
        premium_long_put = self.model(self.S0, strikes[1], r=self.r, T=self.T, N=self.N, div=self.div, sigma=sigmas[1], is_put=True, is_am=True).price()
        long_put = Payoff(self.S0, strikes[1], premium_long_put, is_put=True)
        long_put_payoff = (long_put.calculate_payoff('long',self.sT) * 100 - self.transaction_cost) * num_contract

        #long straddle 
        payoff = long_call_payoff + long_put_payoff 
        current_payoff = ((long_call.current_payoff() + long_put.current_payoff()) * 100 - 2 * self.transaction_cost) * num_contract

        print('# Long Straddle Strategy' if strikes[0] == strikes[1] else '# Long Strangle')
        print('-- Long {} Call:'.format(strikes[0]), long_call)
        print('-- Long {} Put:'.format(strikes[1]), long_put)
        print('-- Delta:', round(self.delta(strikes[0], sigmas[0], False, num_contract) + self.delta(strikes[1], sigmas[1], True, num_contract), 4))
        print('-- Gamma:', round(self.gamma(strikes[0], sigmas[0], False, num_contract) + self.gamma(strikes[1], sigmas[1], True, num_contract), 4))
        print('-- Theta:', round(self.theta(strikes[0], sigmas[0], False, num_contract) + self.theta(strikes[1], sigmas[1], True, num_contract), 4))
        print('-- Vega:', round(self.vega(strikes[0], sigmas[0], False, num_contract) + self.vega(strikes[1], sigmas[1], True, num_contract), 4))
        print('-- Rho:', round(self.rho(strikes[0], sigmas[0], False, num_contract) + self.rho(strikes[1], sigmas[1], True, num_contract), 4))
        print('-- Profit and Loss:', round(current_payoff, 4))
        print('\n')

        _, ax = plt.subplots()
        ax.spines['bottom'].set_position('zero')
        ax.plot(self.sT,long_call_payoff,'--',label='Long Call',color='g')
        ax.plot(self.sT,long_put_payoff,'--',label='Long Put',color='r')
        ax.plot(self.sT,payoff,label='Long Straddle' if strikes[0] == strikes[1] else 'Long Strangle')

        plt.title('Long Straddle' if strikes[0] == strikes[1] else 'Long Strangle')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit and loss')
        plt.legend()
        plt.grid()

        plt.show()
        return current_payoff

    def delta_hedge(self):
        pass

    def put_call_parity(self):
        pass
    
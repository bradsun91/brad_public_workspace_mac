import numpy as np 

from strategies import Strategy
from Binomial_CRR import CRR

S0 = 229.24 # spot price
r = 0.001 # interest rate
T = 28/365 # time to matuity 
N = 10000 # number of time steps
div = 0 # dividend
# num_contract = 1

#stock price range at the end of maturity 
sT = np.arange(0.5*S0, 1.5*S0, 1)

#strategy class initialize
strategy = Strategy(S0, r, T, N, div, sT = sT, model = CRR, is_am = True)

strike_long_call = 240
strike_short_call = 245
strike_long_put = 235
strike_short_put = 230

sigma_long_call = 0.7363
sigma_short_call = 0.7108
sigma_long_put = 0.7294
sigma_short_put = 0.7479

#iron condor
strikes_iron_condor = [strike_long_call, strike_short_call, strike_long_put, strike_short_put]
sigmas_iron_condor = [sigma_long_call, sigma_short_call, sigma_long_put, sigma_short_put]
iron_condor_payoff = strategy.iron_condor(strikes_iron_condor, sigmas_iron_condor)

# # call spread
# strikes_call_spread = [strike_long_call, strike_short_call]
# sigmas_call_spread = [sigma_long_call, sigma_short_call]
# call_spread_payoff = strategy.call_spread(strikes_call_spread, sigmas_call_spread)

# # put spread 
# strikes_put_spread = [strike_long_put, strike_short_put]
# sigmas_put_spread = [sigma_long_put, sigma_short_put]
# put_spread_payoff = strategy.put_spread(strikes_put_spread, sigmas_put_spread)

# # long straddle / strangle
# strikes_long_straddle = [strike_long_call, strike_long_put]
# sigmas_long_straddle = [sigma_long_call, sigma_long_put]
# long_straddle_payoff = strategy.long_straddle(strikes_long_straddle, sigmas_long_straddle)

# naked call / put
long_call_payoff = strategy.naked_call(strike_long_call, sigma_long_call, 'long')
# long_put_payoff = strategy.naked_put(strike_long_put, sigma_long_put, 'long')
# short_call_payoff = strategy.naked_call(strike_short_call, sigma_short_call, 'short')
# short_put_payoff = strategy.naked_put(strike_short_put, sigma_short_put, 'short')
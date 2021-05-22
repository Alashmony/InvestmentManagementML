import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as plt

def drawdown(ret_ser: pd.Series):
    """
    Lets Calculate it:
    1. Compute wealth index
    2. Compute previous peaks
    3. Compute Drawdown - which is the wealth value as a percentage of the previous peak
    """
    wealth_index = 1000*(1+ret_ser).cumprod()
    prev_peak = wealth_index.cummax()
    draw_down = (wealth_index-prev_peak)/prev_peak
    return pd.DataFrame({
        "Wealth Index": wealth_index,
        "Previous Peak": prev_peak,
        "Drawdown" : draw_down        
    })


def all_pfme():
    """
    This Function reads all data in the Portfolios_Formed_on_ME_monthly_EW file.
    """
    pfme_df = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv", index_col=0 , na_values=-99.99, parse_dates= True)
    pfme_df.index = pd.to_datetime(pfme_df.index, format="%Y%m")
    pfme_df.index = pfme_df.index.to_period('M')
    pfme_df = pfme_df/100
    return pfme_df

def get_ffme_returns():
    """
    This Function only reads the Large Cap and Small Cap (Hi 10 and Lo 10) in the Portfolios_Formed_on_ME_monthly_EW file.
    """
    rets = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv", index_col=0 , na_values=-99.99, parse_dates= True)
    rets = rets[['Lo 10','Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets.index = pd.to_datetime(rets.index, format="%Y%m")
    rets.index = rets.index.to_period('M')
    rets = rets/100
    return rets

def get_hfi_returns():
    """
    This Function reads Hedge Fund indecies only
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv", index_col=0 , na_values=-99.99, parse_dates= True)
    hfi.index = hfi.index.to_period('M')
    hfi = hfi/100
    return hfi


def semideviation(r):
    """
    Returns the semi-deviation (Negative Deviation) , 
    
 
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    q_sig = (sigma_r**3)
    skw = exp/q_sig
    return skw
def skewness(r):
    """
    Alternative to Scipy Skewness (scipy.skew()), 
    This one calculate Skewness of a Series or DF and return Float or Series of Floats
    Calculation depends on population STD (N) not sample STD (n-1) 
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    q_sig = (sigma_r**3)
    skw = exp/q_sig
    return skw

def kurt(r):
    """
    Alternative to Scipy Kurtosis (scipy.kurtosis()), 
    This one calculate Kurtosis of a Series or DF and return Float or Series of Floats
    Calculation depends on population STD (N) not sample STD (n-1) 
    The Kurtosis is not the variance of normality, to calculate variance justify it with -3 or justify scipy with +3 
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    q_sig = (sigma_r**4)
    k = exp/q_sig
    return k

def is_normal(r , level = 0.1):
    """
    Applies Jarque-Bera test to determine if the dataseries is normally distributed
    Test applied with default value of 1%
    Returns True if hypothesis of being normal accepted
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value > level

def gbm(n_years =10, n_scenarios = 1000, mu=0.07,sigma = 0.15, steps_per_year = 12, s_0 = 100.0):
    """
    Evolution of a Stock Price using Geometric Browian Motion Model (Monte Carlo Simulation)
    """
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    rets_plus_1 = np.random.normal(loc= (1+mu*dt),scale = (sigma*np.sqrt(dt)),size = (n_steps, n_scenarios))
    rets_plus_1[0] = 1
    prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
    return prices

def show_gbm(n_scenarios, mu, sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    s_0=100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color="indianred", alpha = 0.5, linewidth=2, figsize=(12,5))
    ax.axhline(y=100, ls=":", color="black")
    # draw a dot at the origin
    ax.plot(0,s_0, marker='o',color='darkred', alpha=0.2)

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = erk.run_cppi(risky_r=pd.DataFrame(risky_r),riskfree_rate=riskfree_rate,m=m, start=start, floor=floor)
    wealth = btr["Wealth"]
    # calculate terminal wealth stats
    y_max=wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    # Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
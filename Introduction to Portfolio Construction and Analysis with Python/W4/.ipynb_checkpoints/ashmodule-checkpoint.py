import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as plt
from scipy.stats import norm
from scipy.optimize import minimize
import ipywidgets as widgets
from IPython.display import display
import math



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
        "Wealth": wealth_index,
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

def get_hfi_returns(other_file = False, denominator = 100,file_path = "data/edhec-hedgefundindices.csv",
                    index_col = 0, na_valus = -99.99 , parse_dates = True, date_to_period = True):
    """
    This Function reads Hedge Fund indecies or any other file
    just make sure to enter file name and path and the denominator if you choosed to read other file
    """
    #check if we need other file
    if other_file and file_path == "data/edhec-hedgefundindices.csv":
        file_path = input("Please enter file name and path:")
    #loop to find if this file can be read in pandas
    file_check = False
    while file_check == False:
        try:
            pd.read_csv(file_path, index_col=index_col , na_values=na_valus, parse_dates= parse_dates)
            print("File Accepted")
            file_check = True
        except:
            print("Error, Please check the file name, path, and format")
            file_path = input("Please reenter file name and path:")
    #check if we need new denominator
    if other_file and denominator == 100:
        denominator =  input("Please enter the denominator:")
    #check if the denominator is integer
    deno_check = False
    while deno_check == False:
        try:
            int(denominator)
            print("Denominator Accepted")
            deno_check = True
        except:
            denominator =  input("Error, Please reenter the denominator (Only numbers and integers are acceptable):")
 
    hfi = pd.read_csv(file_path, index_col=index_col , na_values=na_valus, parse_dates= parse_dates)
    if date_to_period: hfi.index = pd.to_datetime(hfi.index, format="%Y%m").to_period('M')
    hfi = hfi/int(denominator)
    hfi.columns =hfi.columns.str.strip() 

    return hfi


def get_idx_returns():
    """
    This Function reads Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", index_col=0 , na_values=-99.99, parse_dates= True)/100
    
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns =ind.columns.str.strip() 
    return ind

def get_total_market_index_returns():
    idx_return = get_idx_returns()
    idx_nfirms = get_hfi_returns(other_file = True, file_path="data/ind30_m_nfirms.csv",denominator = 1)
    idx_size = get_hfi_returns(other_file = True, file_path="data/ind30_m_size.csv",denominator = 1)
    ind_mktcap = idx_nfirms * idx_size
    total_mktcap = ind_mktcap.sum(axis ="columns")
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = "rows")
    ind_capweight.columns = ind_capweight.columns.str.strip()
    total_market_return = (ind_capweight * idx_return).sum(axis = "columns")
    total_market_index = pd.DataFrame(drawdown(total_market_return)["Wealth"])
    return total_market_return

def semideviation(r):
    """
    Returns the semi-deviation (Negative Deviation) of r , 
    r must be a Series or DataFrame
 
    """
    is_negative = r <0
    return r [is_negative].std(ddof=0)

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

def kurtosis(r):
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



""" This part is pure from the code by EDHEC """
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


''' End of pure EDHEC code '''

def is_normal(r , level = 0.1):
    """
    Applies Jarque-Bera test to determine if the dataseries is normally distributed
    Test applied with default value of 1%
    Returns True if hypothesis of being normal accepted
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level =5):
    """
    Returns the historic Value at Risk at a specified level
    i.e.returns the number such that "Level" percent of returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def var_gaussian(r,level=5, modified = False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame 
    If modified is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        #modify the Z Score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z+
                (z**2 -1)      * s/6         +
                (z**3 -3*z)    * (k-3)/24    -
                (2*z**3 - 5*z) * (s**2)/36
            )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level =5):
    """
    Returns the Conditional Value at Risk at a specified level
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r,level= level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov,style =".-",figsize=(15,7),title= "The Effiecient Fronteir",color='C0'):
    """
    Plots the 2-Asset Effecient Frontier
    """
    if er.shape[0] !=2 or er.shape[0]!=2:
        raise ValueError("plot_ef2 only plot 2 asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w , cov) for w in weights]
    ef = pd.DataFrame({
        "Returns" : rets,
        "Volatility" : vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style,figsize=figsize,title=title,color=color)

def minimize_vol(target_return, er, cov):
    """
    Target returns -> Weights
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0,1),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    results = minimize(portfolio_vol, init_guess,
                       args = (cov) , method = "SLSQP",
                       options = {'disp': False},
                       constraints = (return_is_target,weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def optimal_weights(n_points, er,cov):
    """
    Returns a list of weights to run the optimizer on to minimize the volatility
    """
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0,1),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    def negative_sharpe_ratio(weights, riskfree_rate,er ,cov):
        """
        Returns the negative value of sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    
    results = minimize(negative_sharpe_ratio, init_guess,
                       args = (riskfree_rate, er, cov) , method = "SLSQP",
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x 

def gmv(cov):
    """
    Returns the weights of thr Global Minimum Vol portfolio
    given the covarience matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1,n),cov)

def plot_ef(n_points, er, cov,style =".-",figsize=(15,7),riskfree_rate = 0, title= "The Effiecient Fronteir",color='C0',
            show_cml = False, cml_markersize = 12, cml_linewidth = 1, cml_linecolor = "green", cml_linestyle = "dashed", cml_marker = 'o', 
            show_ew=False, ew_color = "goldenrod", ew_marker = "o", ew_markersize = 12,
            show_gmv= False,gmv_color = "midnightblue", gmv_marker = "o", gmv_markersize = 12):
    """
    Plots the Multi-Asset Effecient Frontier, Equally weighted portfolio, and 
    """
    weights = optimal_weights(n_points, er , cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w , cov) for w in weights]
    ef = pd.DataFrame({
        "Returns" : rets,
        "Volatility" : vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style,figsize=figsize,title=title,color=color)
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er , cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        title = "The Effiecient Fronteir and Capital Market Line"
        #CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = cml_linecolor, marker = cml_marker, linestyle = cml_linestyle, markersize = cml_markersize,linewidth = cml_linewidth)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n , n)
        r_ew = portfolio_return(w_ew , er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display the point
        ax.plot([vol_ew], [r_ew],color = ew_color, marker = ew_marker, markersize = ew_markersize)
    if show_gmv:
        n = er.shape[0]
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv , er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display the point
        ax.plot([vol_gmv], [r_gmv],color = gmv_color, marker = gmv_marker, markersize = gmv_markersize)
    
    return ax 


def run_cppi(risky_r, safe_r = None, m=3, start = 1000, floor = 0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI Strategy, given a set of returns for the risky asset
    Retrns a dictionary containing: Asset Value History, Risk Budget History, Risky weight history 
    """
    
    # Setup the CPPI Parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start
    
    if isinstance(risky_r,pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ["R"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12
    # Setup some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)
    
    # Set the CPPI Algorithm
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        #print(cushion)
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        #print(risky_w)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # Compute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        #print(risky_alloc)
        #print(account_value)
        # Save to the DataFrame for Analysis and Plotting purposes
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    
    risky_wealth = start*(1+risky_r).cumprod() 
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": risky_w_history,
        "m":m,
        "start": start,
        "floor":floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "Cushion History":cushion_history,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
        
    return backtest_result

def summary_stats(r, riskfree_rate = 0.03, periods_per_year = 12, level = 5):
    """
    Returns a DataFrame that contains aggrgated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(lambda r : drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    var = r.aggregate(var_gaussian, level = level)
    cf_var = r.aggregate(var_gaussian,level = level, modified = True)
    hist_cvar = r.aggregate(cvar_historic, level = level)
    
    sum_res = pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "VaR "+str(level)+"%":var,
        "Cornish-Fisher VaR "+str(level)+"%":cf_var,
        "Historic CVaR "+str(level)+"%":hist_cvar,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd      
    })

    return sum_res



def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val



def show_gbm(n_scenarios, mu, sigma,s_0=100):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
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
    
def discount(t,r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts


def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()


def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets, r) / pv(liabilities, r)

def inst_to_ann(r):
    """
    Converts short rate to an annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts annualized to a short rate
    """
    return np.log1p(r)

def ear(r=.12 ,N=1):
    """
    Computes the effective annual rate for the rate r paid N times per year 
    """
    return (1+r/N)**N

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate,
                                       coupons_per_year, discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate,coupons_per_year = 1):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discount_rate = discount_rate/coupons_per_year
    
    discounted_flows = erk.discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return (np.average(flows.index, weights=weights.iloc[:,0]))/coupons_per_year

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

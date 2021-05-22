import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as plt
from scipy.stats import norm
from scipy.optimize import minimize


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


def get_idx_returns():
    """
    This Function reads Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", index_col=0 , na_values=-99.99, parse_dates= True)/100
    
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns =ind.columns.str.strip() 
    return ind

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



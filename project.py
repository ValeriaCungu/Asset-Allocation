# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pandas
# !pip install seaborn
# !pip install numpy
# !pip install matplotlib
# !pip install scipy
# !pip install cmake
# !pip install scikit-learn
# !pip install PyPortfolioOpt

# To ignore because they have NaN values at the beginning of the evaluation
to_ignore = [
    #INCOMPLETE
    'PIRELLI & C',
    'ILLIMITY BANK',
    'AQUAFIL',
    'EQUITA GROUP',
    'ITALGAS',
    'ENAV',
    'POSTE ITALIANE',
    'GAMBERO ROSSO',
    # DELISTED
    'BORGOSESIA RSP DEAD - DELIST.28/07/21',
    'BANCA INTERMOBILIARE DEAD - DELIST.29/04/22',
    'CATTOLICA ASSICURAZIONI DEAD - DELIST.12/08/22',
    'ASTALDI DEAD - DELIST.02/08/21',
    'DEA CAPITAL DEAD - DELIST.08/03/23'
]

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.ticker import FixedLocator, FixedFormatter
import random
from scipy.stats import jarque_bera
from pypfopt import expected_returns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg
from collections import OrderedDict
from pypfopt import risk_models

# Define the file path
file_path = 'sample_data/data for exam 2023.xlsx'

# Read the Excel file and select the 'stocks daily' sheet
df_stocks_daily = pd.read_excel(file_path, sheet_name='stocks daily').drop(to_ignore, axis=1)
df_stocks_monthly = pd.read_excel(file_path, sheet_name='stocks monthly').drop(to_ignore, axis=1)
df_funds_monthly = pd.read_excel(file_path, sheet_name='funds monthly')

def reformat_df(df):
    # Copy dataframe
    copy = df.copy()
    # Remove the second row
    copy.drop([0,1], inplace=True)
    # Reset the index of the DataFrame
    copy.reset_index(drop=True, inplace=True)
    copy.rename(columns = {'Name':'Date'}, inplace = True)
    copy['Date'] = pd.to_datetime(copy['Date'])
    return copy.set_index('Date')

def weighted_mean(numbers, weights):
  sum_of_products = 0
  sum_of_weights = 0
  for number, weight in zip(numbers, weights):
    sum_of_products += number * weight
    sum_of_weights += weight

  return sum_of_products / sum_of_weights

def get_only_data(df):
    return df.iloc[:, 1:]
def get_only_date(df):
    return df.iloc[:, 0]

def get_stats_concepts(df):
    return pd.DataFrame({
        'Mean': df.mean(),
        'Standard Deviation': df.std(),
        'Variance': df.var(),
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis()
    })

def plot_heatmap(title, df):
    pyplot.figure(figsize=(16, 16))
    pyplot.title(title)
    #plotting the heatmap for correlation
    return sns.heatmap(df, xticklabels=True, yticklabels=True)

"""# 1. Focus first on the two worksheet on stocks. Compute returns for both daily and monthly stocks. Compute mean, standard deviation, variance, skewness and kurtosis for stocks at daily and monthly frequency. Show the results in a table and comment."""

daily = reformat_df(df_stocks_daily)
daily

monthly = reformat_df(df_stocks_monthly)
monthly

# Calculate daily returns for each stock
returns_daily = get_only_data(daily).pct_change()
# Compute statistics for daily returns
daily_stats = get_stats_concepts(returns_daily)
# Calculate monthly returns for each stock
returns_monthly = get_only_data(monthly).pct_change()
# Compute statistics for monthly returns
monthly_stats = get_stats_concepts(returns_monthly)
# Create a table to display the results
table_daily_monthly_stats = pd.concat([daily_stats,monthly_stats], axis=1, keys=['Daily','Monthly'])
# Print the results
table_daily_monthly_stats

def get_max_min(df, column_name, sample_rate):
    sort_df = df.sort_values(column_name, ascending=False)
    name_max = sort_df.iloc[0].name
    value_max = sort_df.iloc[0][column_name]
    name_min = sort_df.iloc[-1].name
    value_min = sort_df.iloc[-1][column_name]

    print(f'MAX - {column_name:20} - {sample_rate:10}: {name_max:50} {round(value_max,5)}')
    print(f'MIN - {column_name:20} - {sample_rate:10}: {name_min:50} {round(value_min,5)}')


get_max_min(daily_stats, 'Mean', 'Daily')
get_max_min(daily_stats, 'Standard Deviation', 'Daily')
get_max_min(daily_stats, 'Variance', 'Daily')
get_max_min(daily_stats, 'Skewness', 'Daily')
get_max_min(daily_stats, 'Kurtosis', 'Daily')

get_max_min(monthly_stats, 'Mean', 'Monthly')
get_max_min(monthly_stats, 'Standard Deviation', 'Monthly')
get_max_min(monthly_stats, 'Variance', 'Monthly')
get_max_min(monthly_stats, 'Skewness', 'Monthly')
get_max_min(monthly_stats, 'Kurtosis', 'Monthly')

def plot_histogram(df, column_name, title):
    array_filtered = df.filter(items=[column_name])[column_name].to_numpy()
    pyplot.title(title)
    ax = sns.histplot(array_filtered, kde=True)
    pyplot.xticks(np.arange(min(array_filtered), max(array_filtered), (max(array_filtered)-min(array_filtered))/20), rotation = 90)
    pyplot.show()

plot_histogram(df=daily_stats, column_name='Mean', title='Daily mean distribution')
plot_histogram(df=monthly_stats, column_name='Mean', title='Monthly mean distribution')

# Compute statistics for log(returns+1)
daily_stats_log = get_stats_concepts(np.log(returns_daily+1))
monthly_stats_log = get_stats_concepts(np.log(returns_daily+1))
plot_histogram(df=daily_stats_log, column_name='Mean', title='Daily mean distribution - log')
plot_histogram(df=monthly_stats_log, column_name='Mean', title='Monthly mean distribution - log')

for column in returns_daily.columns:
    data= returns_daily.loc[:,column].to_numpy()
    statistic,pvalue=jarque_bera(data, nan_policy='omit')
    if pvalue >= 0.05:
        print(f'{column:30} - {round(pvalue,3)}')
print('END')

for column in returns_daily.columns:
    data= returns_daily.loc[:,column].to_numpy()
    statistic,pvalue=jarque_bera(data, nan_policy='omit')
    print(f'{column:50} - {round(pvalue,3)}')

for column in returns_monthly.columns:
    data= returns_monthly.loc[:,column].to_numpy()
    statistic,pvalue=jarque_bera(data, nan_policy='omit')
    if pvalue >= 0.05:
        print(f'{column:30}  {round(pvalue,3)}')
print('END')

for column in returns_monthly.columns:
    data= returns_monthly.loc[:,column].to_numpy()
    statistic,pvalue=jarque_bera(data, nan_policy='omit')
    print(f'{column:50} - {round(pvalue,3)}')

"""# 2. Compute the variance-covariance matrix and the correlation matrix.'"""

daily_without_ignore_columns = returns_daily.copy()
monthly_without_ignore_columns = returns_monthly.copy()
daily_variance_covariance = daily_without_ignore_columns.astype(float).cov()
monthly_variance_covariance = monthly_without_ignore_columns.astype(float).cov()
daily_correlation = daily_without_ignore_columns.astype(float).corr()
monthly_correlation = monthly_without_ignore_columns.astype(float).corr()

ax = plot_heatmap(title='Variance-Covariance - Daily', df=daily_variance_covariance)

ax = plot_heatmap(title='Variance-Covariance - Monthly', df=monthly_variance_covariance)

ax = plot_heatmap(title='Correlation - Daily', df=daily_correlation)

ax = plot_heatmap(title='Correlation - Monthly', df=monthly_correlation)

def get_less_correlated_pair_stocks(correlation_matrix):
    matrix_correlation_returns_sorted = convert_upper_diagonal_to_nan(correlation_matrix.copy()).unstack().sort_values().to_frame().reset_index()
    selected_columns = []
    for index, row in matrix_correlation_returns_sorted.iterrows():
        if row['level_0'] not in selected_columns:
            selected_columns.append(row['level_0'])
            if len(selected_columns) == 20:
                break
        if row['level_1'] not in selected_columns:
            selected_columns.append(row['level_1'])
            if len(selected_columns) == 20:
                break
    return selected_columns


def get_colleration_pair_stocks(correlation_matrix_stacked, stock_name_1, stock_name_2):
    return correlation_matrix_stacked.loc[(correlation_matrix_stacked['level_0'] == stock_name_1) & (correlation_matrix_stacked['level_1'] == stock_name_2)].iloc[0, 2]

def get_less_correlated_pair_stocks_new(correlation_matrix, preselected_stocks = [], n_stock_to_select = 10):
    matrix_correlation_returns_sorted = correlation_matrix.copy().unstack().sort_values().to_frame().dropna().reset_index()

    for threshold_base in range(1, 200):
        threshold = threshold_base/100
        stocks_ordered_by_correlation = []

        for index, row in matrix_correlation_returns_sorted.iterrows():
            stocks_ordered_by_correlation.append(row['level_0'])
            stocks_ordered_by_correlation.append(row['level_1'])


        stocks_ordered_by_correlation = list(dict.fromkeys(stocks_ordered_by_correlation))
        selected_columns = preselected_stocks
        for stock in stocks_ordered_by_correlation:
            if len(selected_columns) == n_stock_to_select:
                print(f'Threshold: {threshold}')
                print(f'Total stocks: {len(selected_columns)}')
                return selected_columns


            if stock in selected_columns:
                continue

            count = len(selected_columns)
            for already_selected_stock in selected_columns:
                if get_colleration_pair_stocks(matrix_correlation_returns_sorted, stock_name_1=stock, stock_name_2=already_selected_stock) < threshold:
                    count -= 1

            if count == 0:
                selected_columns.append(stock)

n_stock = 10

less_correlated_pair_stocks_daily_new = get_less_correlated_pair_stocks_new(daily_correlation, preselected_stocks = [], n_stock_to_select = n_stock)
less_correlated_pair_stocks_daily_new

"""### Average daily returns of daily selected"""

pd.DataFrame(daily, columns = less_correlated_pair_stocks_daily_new).pct_change().mean().mean()*100

"""### Average monthly returns of daily selected"""

pd.DataFrame(monthly, columns = less_correlated_pair_stocks_daily_new).pct_change().mean().mean()*100

less_correlated_pair_stocks_monthly_new =  get_less_correlated_pair_stocks_new(monthly_correlation, preselected_stocks = [], n_stock_to_select = n_stock)
less_correlated_pair_stocks_monthly_new

"""### Average daily returns of monthly selected"""

pd.DataFrame(daily, columns = less_correlated_pair_stocks_monthly_new).pct_change().mean().mean()*100

"""### Average monthly returns of monthly selected"""

pd.DataFrame(monthly, columns = less_correlated_pair_stocks_monthly_new).pct_change().mean().mean()*100

"""# 3. Select a sample made of 10-12 securities. You should motivate your choice of securities. The choice can be made, for example, on the basis of the correlation structure of the variance-covariance matrix. Explain and justify your choices."""

def plot_stocks(df, title, columns):
    ax = df.plot(figsize=(10, 8))
    ax.grid(which = "major", linewidth = 1, alpha=0.3)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.3)
    ax.minorticks_on()
    ax.set_title(title,color='black')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount (€)')

    ax.legend(bbox_to_anchor=(1.0, 1.0), frameon=False)

    # Modify y-axis tick labels with symbol
    ticks = ax.get_yticks()
    labels = [f'€{tick}' for tick in ticks]

    # Set modified labels and fixed locator on the y-axis
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))

# get random stock columns
samples = 10
columns = [col for col in daily.columns if col != 'Date']
selected_columns = less_correlated_pair_stocks_monthly_new

# select random stock
selected_daily = daily.filter(items=selected_columns)
selected_monthly = monthly.filter(items=selected_columns)

"""# 4. Plot the behavior of the security prices you have chosen, both in daily and monthly frequency during the entire lenght of the sample size."""

ax = selected_daily.plot(figsize=(10, 8))

ax = selected_monthly.plot(figsize=(10, 8))

"""# 5. Compute the Mean Variance optimal portfolio allocation for the sample of securities chosen by you both in daily and monthly frequency. Discuss."""

def plot_weights(weights, title):
    # Plotting the values
    x = np.array(list(weights.keys()))
    y = np.array(list(weights.values()))*100

    ax = pyplot.bar(x, y)
    pyplot.suptitle(title, fontsize=12)

    pyplot.ylabel('Weights')
    pyplot.xticks(rotation=30, ha='right')

    for i in range(len(x)):
        pyplot.text(i,y[i] // 2,f"{round(y[i],2)} %", ha = 'center',
                     bbox = dict(facecolor = 'white'))

    pyplot.show()

def plot_weights_arr(weights_arr = [], title_arr = []):
    num_plots = len(weights_arr)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    fig, axes = pyplot.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num_plots):
        axes[i].set_title(title_arr[i])
        axes[i].set_ylabel('Weights')

        keys = list(weights_arr[i].keys())
        values = np.array(list(weights_arr[i].values())) * 100

        bars = axes[i].bar(keys, values)
        axes[i].set_xticks(axes[i].get_xticks(),axes[i].get_xticklabels(), rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height // 2),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom', bbox = dict(facecolor = 'white'), fontsize=8)

    # Remove any extra empty subplots
    if num_plots < len(axes):
        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

    pyplot.tight_layout()
    pyplot.show()

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.risk_models import sample_cov

def get_mu_s(prices, stocks_names, frequency):
    mu = mean_historical_return(prices.filter(items=stocks_names), frequency=frequency)
    S = sample_cov(prices.filter(items=stocks_names), frequency=frequency)
    return (mu, S)

def get_efficient_frontier(mu,s,allow_short_selling):
    weight_bounds = (0,1)
    if allow_short_selling is True:
        weight_bounds = (-1,1)
    else:
        weight_bounds = (0,1)
    return EfficientFrontier(mu, s , weight_bounds=weight_bounds)

def portfolio_weighted(portfolio_returns, df_returns, df_weights):
    for col in df_returns.columns:
        for i in range(0,len(df_returns)):
            df_returns.iloc[i][col] = portfolio_returns.iloc[i][col]*df_weights.loc[col]["Weight"]
    return(df_returns)

"""## MONTHLY"""

portfolio_monthly_avg_returns = expected_returns.mean_historical_return(selected_monthly)
portfolio_monthly_avg_returns.round(4)

portfolio_monthly_cov_mat = risk_models.sample_cov(selected_monthly)
portfolio_monthly_cov_mat.round(4)

portfolio_monthly_returns = returns_monthly.filter(items=selected_columns)
mu_monthly = portfolio_monthly_returns.mean()
sigma_monthly = portfolio_monthly_returns.cov()

# Portfolio Optimization - Monthly data:  minimum volatility (non-negative)
ef_min_volatility_monthly = EfficientFrontier(mu_monthly, sigma_monthly, weight_bounds=(0, 1))
weights_min_volatility_monthly = ef_min_volatility_monthly.min_volatility()
cleaned_weights_min_volatility_monthly = pd.DataFrame(ef_min_volatility_monthly.clean_weights().values(),
                                        index = portfolio_monthly_returns.columns, columns=["Weight"])

# Portfolio Optimization - Monthly data:  minimum volatility
ef_min_volatility_shorting_monthly = EfficientFrontier(mu_monthly, sigma_monthly, weight_bounds=(-1, 1))
weights_min_volatility_shorting_monthly = ef_min_volatility_shorting_monthly.min_volatility()
cleaned_weights_min_volatility_shorting_monthly = pd.DataFrame(ef_min_volatility_shorting_monthly.clean_weights().values(),
                                         index = portfolio_monthly_returns.columns, columns=["Weight"])

# Portfolio Optimization - Monthly data:  maximal Sharpe ratio (a.k.a the tangency portfolio, non-negative)
ef_max_sharpe_ratio_monthly = EfficientFrontier(mu_monthly, sigma_monthly, weight_bounds=(0, 1))
weights_max_sharpe_ratio_monthly = ef_max_sharpe_ratio_monthly.max_sharpe(risk_free_rate=0)
cleaned_weights_max_sharpe_ratio_monthly = pd.DataFrame(ef_max_sharpe_ratio_monthly.clean_weights().values(),
                                        index = portfolio_monthly_returns.columns, columns=["Weight"])

# Portfolio Optimization - Monthly data:  maximal Sharpe ratio (a.k.a the tangency portfolio)
ef_max_sharpe_ratio_shorting_monthly = EfficientFrontier(mu_monthly, sigma_monthly, weight_bounds=(-1, 1))
weights_max_sharpe_ratio_shorting_monthly = ef_max_sharpe_ratio_shorting_monthly.max_sharpe(risk_free_rate=0)
cleaned_weights_max_sharpe_ratio_shorting_monthly = pd.DataFrame(ef_max_sharpe_ratio_shorting_monthly.clean_weights().values(),
                                         index = portfolio_monthly_returns.columns, columns=["Weight"])


plot_weights_arr(
    weights_arr=[
        weights_min_volatility_monthly,
        weights_min_volatility_shorting_monthly,
        weights_max_sharpe_ratio_monthly,
        weights_max_sharpe_ratio_shorting_monthly
        ],
    title_arr=[
        'Monthly - not allowing short-selling - Minimum volatility',
        'Monthly - allowing short-selling - Minimum volatility',
        'Monthly - not allowing short-selling - Max sharpe ratio',
        'Monthly - allowing short-selling - Max sharpe ratio',
        ]
)

# Portfolio Monthly - not allowing short-selling - Minimum volatility
portfolio_monthly_returns_min_volatility = pd.DataFrame(index = portfolio_monthly_returns.index,
                                          columns = portfolio_monthly_returns.columns)
portfolio_monthly_returns_min_volatility = portfolio_weighted(portfolio_monthly_returns, portfolio_monthly_returns_min_volatility, cleaned_weights_min_volatility_monthly)


# Portfolio Monthly - allowing short-selling - Minimum volatility
portfolio_monthly_returns_min_volatility_shorting = pd.DataFrame(index = portfolio_monthly_returns.index,
                                           columns = portfolio_monthly_returns.columns)
portfolio_monthly_returns_min_volatility_shorting = portfolio_weighted(portfolio_monthly_returns, portfolio_monthly_returns_min_volatility_shorting, cleaned_weights_min_volatility_shorting_monthly)


# Portfolio Monthly - not allowing short-selling - Max sharpe ratio
portfolio_monthly_returns_max_sharpe_ratio = pd.DataFrame(index = portfolio_monthly_returns.index,
                                           columns = portfolio_monthly_returns.columns)
portfolio_monthly_returns_max_sharpe_ratio = portfolio_weighted(portfolio_monthly_returns, portfolio_monthly_returns_max_sharpe_ratio, cleaned_weights_max_sharpe_ratio_monthly)


# Portfolio Monthly - allowing short-selling - Max sharpe ratio
portfolio_monthly_returns_max_sharpe_ratio_shorting = pd.DataFrame(index = portfolio_monthly_returns.index,
                                           columns = portfolio_monthly_returns.columns)
portfolio_monthly_returns_max_sharpe_ratio_shorting = portfolio_weighted(portfolio_monthly_returns, portfolio_monthly_returns_max_sharpe_ratio_shorting, cleaned_weights_max_sharpe_ratio_shorting_monthly)

np.random.seed(42)
init_ret = portfolio_monthly_returns
num_ports = 3000 # number of portfolios
all_weights = np.zeros((num_ports, len(init_ret.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    # Weights
    weights = np.array(np.random.random(len(init_ret.columns)))
    weights = weights/np.sum(weights)

    # Save weights
    all_weights[x,:] = weights

    # Expected return
    ret_arr[x] = np.sum( (init_ret.mean() * weights))

    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(init_ret.cov(), weights)))

    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]

ef_max_sharpe_ratio_shorting_monthly.portfolio_performance()

portfolio_monthly_returns_max_sharpe_ratio_shorting.mean().mean()

pyplot.figure(figsize=(12,8))
pyplot.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
pyplot.colorbar(label='Sharpe Ratio')
pyplot.xlabel('Volatility')
pyplot.ylabel('Return')
pyplot.show()

"""## DAILY

"""

portfolio_daily_avg_returns = expected_returns.mean_historical_return(selected_daily)
portfolio_daily_avg_returns.round(4)

portfolio_daily_cov_mat = risk_models.sample_cov(selected_daily)
portfolio_daily_cov_mat.round(4)

portfolio_daily_returns = returns_daily.filter(items=selected_columns)
mu_daily = portfolio_daily_returns.mean()
sigma_daily = portfolio_daily_returns.cov()

# Portfolio Optimization - Daily data:  minimum volatility (non-negative)
ef_min_volatility_daily = EfficientFrontier(mu_daily, sigma_daily, weight_bounds=(0, 1))
weights_min_volatility_daily = ef_min_volatility_daily.min_volatility()
cleaned_weights_min_volatility_daily = pd.DataFrame(ef_min_volatility_daily.clean_weights().values(),
                                        index = portfolio_daily_returns.columns, columns=["Weight"])

# Portfolio Optimization - Daily data:  minimum volatility
ef_min_volatility_shorting_daily = EfficientFrontier(mu_daily, sigma_daily, weight_bounds=(-1, 1))
weights_min_volatility_shorting_daily = ef_min_volatility_shorting_daily.min_volatility()
cleaned_weights_min_volatility_shorting_daily = pd.DataFrame(ef_min_volatility_shorting_daily.clean_weights().values(),
                                         index = portfolio_daily_returns.columns, columns=["Weight"])

# Portfolio Optimization - Daily data:  maximal Sharpe ratio (a.k.a the tangency portfolio, non-negative)
ef_max_sharpe_ratio_daily = EfficientFrontier(mu_daily, sigma_daily, weight_bounds=(0, 1))
weights_max_sharpe_ratio_daily = ef_max_sharpe_ratio_daily.max_sharpe(risk_free_rate=0)
cleaned_weights_max_sharpe_ratio_daily = pd.DataFrame(ef_max_sharpe_ratio_daily.clean_weights().values(),
                                        index = portfolio_daily_returns.columns, columns=["Weight"])

# Portfolio Optimization - Daily data:  maximal Sharpe ratio (a.k.a the tangency portfolio)
ef_max_sharpe_ratio_shorting_daily = EfficientFrontier(mu_daily, sigma_daily, weight_bounds=(-1, 1))
weights_max_sharpe_ratio_shorting_daily = ef_max_sharpe_ratio_shorting_daily.max_sharpe(risk_free_rate=0)
cleaned_weights_max_sharpe_ratio_shorting_daily = pd.DataFrame(ef_max_sharpe_ratio_shorting_daily.clean_weights().values(),
                                         index = portfolio_daily_returns.columns, columns=["Weight"])


plot_weights_arr(
    weights_arr=[
        weights_min_volatility_daily,
        weights_min_volatility_shorting_daily,
        weights_max_sharpe_ratio_daily,
        weights_max_sharpe_ratio_shorting_daily
        ],
    title_arr=[
        'Daily - not allowing short-selling - Minimum volatility',
        'Daily - allowing short-selling - Minimum volatility',
        'Daily - not allowing short-selling - Max sharpe ratio',
        'Daily - allowing short-selling - Max sharpe ratio',
        ]
)



# Portfolio Daily - not allowing short-selling - Minimum volatility
portfolio_daily_returns_min_volatility = pd.DataFrame(index = portfolio_daily_returns.index,
                                          columns = portfolio_daily_returns.columns)
portfolio_daily_returns_min_volatility = portfolio_weighted(portfolio_daily_returns, portfolio_daily_returns_min_volatility, cleaned_weights_min_volatility_daily)


# Portfolio Daily - allowing short-selling - Minimum volatility
portfolio_daily_returns_min_volatility_shorting = pd.DataFrame(index = portfolio_daily_returns.index,
                                           columns = portfolio_daily_returns.columns)
portfolio_daily_returns_min_volatility_shorting = portfolio_weighted(portfolio_daily_returns, portfolio_daily_returns_min_volatility_shorting, cleaned_weights_min_volatility_shorting_daily)


# Portfolio Daily - not allowing short-selling - Max sharpe ratio
portfolio_daily_returns_max_sharpe_ratio = pd.DataFrame(index = portfolio_daily_returns.index,
                                           columns = portfolio_daily_returns.columns)
portfolio_daily_returns_max_sharpe_ratio = portfolio_weighted(portfolio_daily_returns, portfolio_daily_returns_max_sharpe_ratio, cleaned_weights_max_sharpe_ratio_daily)


# Portfolio Daily - allowing short-selling - Max sharpe ratio
portfolio_daily_returns_max_sharpe_ratio_shorting = pd.DataFrame(index = portfolio_daily_returns.index,
                                           columns = portfolio_daily_returns.columns)
portfolio_daily_returns_max_sharpe_ratio_shorting = portfolio_weighted(portfolio_daily_returns, portfolio_daily_returns_max_sharpe_ratio_shorting, cleaned_weights_max_sharpe_ratio_shorting_daily)

np.random.seed(42)
init_ret = portfolio_daily_returns
num_ports = 3000 # number of portfolios
all_weights = np.zeros((num_ports, len(init_ret.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    # Weights
    weights = np.array(np.random.random(len(init_ret.columns)))
    weights = weights/np.sum(weights)

    # Save weights
    all_weights[x,:] = weights

    # Expected return
    ret_arr[x] = np.sum( (init_ret.mean() * weights))

    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(init_ret.cov(), weights)))

    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]

ef_max_sharpe_ratio_shorting_daily.portfolio_performance()

portfolio_daily_returns_max_sharpe_ratio_shorting.mean().mean()

pyplot.figure(figsize=(12,8))
pyplot.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
pyplot.colorbar(label='Sharpe Ratio')
pyplot.xlabel('Volatility')
pyplot.ylabel('Return')
pyplot.show()

from pypfopt.plotting import plot_efficient_frontier

def efficient_frontier(mu, S, weight_bounds, title):
  ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
  fig, ax = pyplot.subplots()
  ef_max_sharpe = ef.deepcopy()
  plot_efficient_frontier(ef, ax=ax, show_assets=False)
  # Find the tangency portfolio
  ef_max_sharpe.max_sharpe(risk_free_rate=0)
  ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
  ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
  # Generate random portfolios
  n_samples = 10000
  w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
  rets = w.dot(ef.expected_returns)
  stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
  sharpes = rets / stds
  ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

  # Output
  ax.set_title(f"Efficient Frontier portfolios - {title} - Max Sharpe Ratio")
  ax.legend()
  pyplot.tight_layout()
  pyplot.show()


efficient_frontier(mu=mu_monthly, S=sigma_monthly, weight_bounds=(0,1), title='Monthly weight bounds (0,1)')
efficient_frontier(mu=mu_daily, S=sigma_daily, weight_bounds=(0,1), title='Daily weight bounds (0,1)')
efficient_frontier(mu=mu_monthly, S=sigma_monthly, weight_bounds=(-1,1), title='Monthly weight bounds (-1,1)')
efficient_frontier(mu=mu_daily, S=sigma_daily, weight_bounds=(-1,1), title='Daily weight bounds (-1,1)')

from pypfopt.plotting import plot_efficient_frontier

def efficient_frontier_min_vol(mu, S, weight_bounds, title):
  ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
  fig, ax = pyplot.subplots()
  ef_max_sharpe = ef.deepcopy()
  plot_efficient_frontier(ef, ax=ax, show_assets=False)
  # Find the tangency portfolio
  ef_max_sharpe.min_volatility()
  ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
  ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min volatility")
  # Generate random portfolios
  n_samples = 10000
  w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
  rets = w.dot(ef.expected_returns)
  stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
  sharpes = rets / stds
  ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

  # Output
  ax.set_title(f"Efficient Frontier portfolios - {title} - Min volatility")
  ax.legend()
  pyplot.tight_layout()
  pyplot.show()


efficient_frontier_min_vol(mu=mu_monthly, S=sigma_monthly, weight_bounds=(0,1), title='Monthly weight bounds (0,1)')
efficient_frontier_min_vol(mu=mu_daily, S=sigma_daily, weight_bounds=(0,1), title='Daily weight bounds (0,1)')
efficient_frontier_min_vol(mu=mu_monthly, S=sigma_monthly, weight_bounds=(-1,1), title='Monthly weight bounds (-1,1)')
efficient_frontier_min_vol(mu=mu_daily, S=sigma_daily, weight_bounds=(-1,1), title='Daily weight bounds (-1,1)')

def get_stats_portfolio(portfolio_daily, portfolio_monthly):
  df_mean = pd.DataFrame({"Daily" : portfolio_daily.mean(), "Monthly" : portfolio_monthly.mean()})
  df_std = pd.DataFrame({"Daily" : portfolio_daily.std(), "Monthly" : portfolio_monthly.std()})
  df_var = pd.DataFrame({"Daily" : portfolio_daily.var(), "Monthly" : portfolio_monthly.var()})
  df_skew = pd.DataFrame({"Daily" : portfolio_daily.skew(), "Monthly" : portfolio_monthly.skew()})
  df_kurt = pd.DataFrame({"Daily" : portfolio_daily.kurtosis(), "Monthly" : portfolio_monthly.kurtosis()})

  return pd.concat([
      df_mean.mean(),
      df_std.mean(),
      df_var.mean(),
      df_skew.mean(),
      df_kurt.mean()
      ], axis = 1)

portfolio_max_sharpe_ratio_shorting_stats = get_stats_portfolio(portfolio_daily_returns_max_sharpe_ratio_shorting, portfolio_monthly_returns_max_sharpe_ratio_shorting)
portfolio_max_sharpe_ratio_shorting_stats.columns = ["Mean", "Std", "Var", "Skewness", "Kurtosis"]
portfolio_max_sharpe_ratio_shorting_stats.round(5)

portfolio_max_sharpe_ratio_stats = get_stats_portfolio(portfolio_daily_returns_max_sharpe_ratio, portfolio_monthly_returns_max_sharpe_ratio)
portfolio_max_sharpe_ratio_stats.columns = ["Mean", "Std", "Var", "Skewness", "Kurtosis"]
portfolio_max_sharpe_ratio_stats.round(5)

portfolio_min_volatility_shorting_stats = get_stats_portfolio(portfolio_daily_returns_min_volatility_shorting, portfolio_monthly_returns_min_volatility_shorting)
portfolio_min_volatility_shorting_stats.columns = ["Mean", "Std", "Var", "Skewness", "Kurtosis"]
portfolio_min_volatility_shorting_stats.round(5)

portfolio_min_volatility_stats = get_stats_portfolio(portfolio_daily_returns_min_volatility, portfolio_monthly_returns_min_volatility)
portfolio_min_volatility_stats.columns = ["Mean", "Std", "Var", "Skewness", "Kurtosis"]
portfolio_min_volatility_stats.round(5)

print('Min volatility')
pd.concat([portfolio_min_volatility_stats.round(5), portfolio_min_volatility_shorting_stats.round(5)], keys=['Not Shorting','Shorting'])

print('Max sharp ratio')
pd.concat([portfolio_max_sharpe_ratio_stats.round(5), portfolio_max_sharpe_ratio_shorting_stats.round(5)], keys=['Not Shorting','Shorting'])

"""# Punto 9 - 10"""

ftse_daily = pd.read_excel(r'sample_data/Data for CAPM.xlsx', sheet_name=0, index_col=0)
ftse_monthly = pd.read_excel(r'sample_data/Data for CAPM.xlsx', sheet_name=1, index_col=0)
ret_daily = pd.read_excel(r'sample_data/Data for CAPM.xlsx', sheet_name=2, index_col=0)
ret_monthly = pd.read_excel(r'sample_data/Data for CAPM.xlsx', sheet_name=3,index_col=0)

def calculate_beta(security_returns, market_returns):
    covariance = np.cov(security_returns.dropna().values, market_returns.dropna().values.flatten())
    beta = covariance[0,1] / np.var(market_returns.dropna().values.flatten())
    return beta

#daily
betas_daily = []
for column in ret_daily.columns:
    security_returns = ret_daily[column]
    beta = calculate_beta(security_returns, ftse_daily)
    betas_daily.append(beta)

#monthly
betas_monthly = []
for column in ret_monthly.columns:
    security_returns = ret_monthly[column]
    beta = calculate_beta(security_returns, ftse_monthly)
    betas_monthly.append(beta)

annualized_ret_monthly = (ret_monthly.mean()*12).tolist()
annualized_ret_daily = (ret_daily.mean()*252).tolist()
ret_monthly.columns

def remove_m_prefix(array):
    return [string.replace("M ", "") for string in array]

ret_monthly.columns = remove_m_prefix(ret_monthly.columns)

def plot_security_market(market_return, beta_portfolio = 1, risk_free_rate = 0.03, title = '', betas_input = [], stock_names = [], annualized_ret=[], only_portfolio=False):
  expected_return_portfolio = risk_free_rate + beta_portfolio * (market_return - risk_free_rate)
  beta_values = np.linspace(0, 2, 100)
  sml = risk_free_rate + beta_values * (market_return - risk_free_rate)
  pyplot.figure(figsize=(10, 6))
  pyplot.grid(linewidth = 0.5, alpha = 0.25)
  pyplot.plot(beta_values, sml, label=title, color='blue',  alpha=0.3)
  colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black']  # Colors for different stocks
  for i in range(11):  # Change according to number of stocks
    if only_portfolio is True:
      i = 10

    stock_name = ''
    if i == 10:
      stock_name = 'Portfolio'
    else:
      stock_name = stock_names[i]
    pyplot.scatter(betas_input[i], annualized_ret[i], color=colors[i])
    pyplot.text(betas_input[i], annualized_ret[i], f'{stock_name}', verticalalignment='bottom')
  pyplot.scatter(beta_portfolio, expected_return_portfolio, color='green',  marker="*", s=300)
  pyplot.annotate('Market', xy=(beta_portfolio, expected_return_portfolio),  xycoords='data',
             xytext=(0, 20), textcoords='offset points',
             size=13, ha='center', va="center")
  pyplot.xlabel('Beta')
  pyplot.ylabel('Expected Return')
  pyplot.legend()
  pyplot.title(title)
  pyplot.grid(True)
  pyplot.show()

plot_security_market(
    market_return=ftse_monthly.mean().squeeze()*12,
    beta_portfolio = 1,
    risk_free_rate = 0.03,
    title = 'Security Market Line - monthly',
    betas_input = betas_monthly,
    stock_names=ret_monthly.columns,
    annualized_ret=annualized_ret_monthly,
    only_portfolio=False
    )

plot_security_market(
    market_return=ftse_daily.mean().squeeze()*252,
    beta_portfolio = 1,
    risk_free_rate = 0.03,
    title = 'Security Market Line - daily',
    betas_input = betas_daily,
    stock_names=ret_daily.columns,
    annualized_ret=annualized_ret_daily,
    only_portfolio=False
    )

plot_security_market(
    market_return=ftse_monthly.mean().squeeze()*12,
    beta_portfolio = 1,
    risk_free_rate = 0.03,
    title = 'Security Market Line - monthly',
    betas_input = betas_monthly,
    stock_names=ret_monthly.columns,
    annualized_ret=annualized_ret_monthly,
    only_portfolio=True
    )

plot_security_market(
    market_return=ftse_daily.mean().squeeze()*252,
    beta_portfolio = 1,
    risk_free_rate = 0.03,
    title = 'Security Market Line - daily',
    betas_input = betas_daily,
    stock_names=ret_daily.columns,
    annualized_ret=annualized_ret_daily,
    only_portfolio=True
    )

def plot_security_market_two_stocks(market_return, beta_mm = [], ret_mm = [], risk_free_rate = 0.03, beta_portfolio = 1, period = '', stock_names = []):
  expected_return_portfolio = risk_free_rate + beta_portfolio * (market_return - risk_free_rate)
  beta_values = np.linspace(0, 2, 100)
  sml = risk_free_rate + beta_values * (market_return - risk_free_rate)
  pyplot.figure(figsize=(10, 6))
  pyplot.grid(linewidth = 0.5, alpha = 0.25)
  pyplot.plot(beta_values, sml, label=f'Security Market Line - {period} 2 stocks', color='blue', alpha=0.3)
  colors = ['red', 'green']
  for i in range(2):  # Change according to number of stocks
    pyplot.scatter(beta_mm[i], ret_mm[i], color=colors[i])
    pyplot.text(beta_mm[i], ret_mm[i], f'{stock_names[i]}', verticalalignment='bottom')
  pyplot.scatter(beta_portfolio, expected_return_portfolio, color='green',  marker="*", s=300)
  pyplot.annotate('Market', xy=(beta_portfolio, expected_return_portfolio),  xycoords='data',
             xytext=(0, 20), textcoords='offset points',
             size=13, ha='center', va="center")
  pyplot.xlabel('Beta')
  pyplot.ylabel('Expected Return')
  pyplot.legend()
  pyplot.title(f'Security Market Line - {period}')
  pyplot.grid(True)
  pyplot.show()

plot_security_market_two_stocks(
    market_return = ftse_monthly.mean().squeeze()*12,
    beta_mm =[betas_monthly[1], betas_monthly[8]],
    ret_mm = [annualized_ret_monthly[1], annualized_ret_monthly[8]],
    risk_free_rate = 0.03,
    beta_portfolio = 1,
    period = 'monthly',
    stock_names=[ret_monthly.columns[1],ret_monthly.columns[8]]
)

plot_security_market_two_stocks(
    market_return = ftse_daily.mean().squeeze()*252,
    beta_mm = [betas_daily[1], betas_daily[8]],
    ret_mm = [annualized_ret_daily[1], annualized_ret_daily[8]],
    risk_free_rate = 0.03,
    beta_portfolio = 1,
    period = 'daily',
    stock_names=[ret_daily.columns[1],ret_daily.columns[8]]
)

"""# Black Litterman Allocation

## Daily
"""

selected_daily.tail()

#market_prices
ftse_daily = pd.read_excel(r'sample_data/Clean data.xlsx', sheet_name='FTSE Daily ', index_col=0).iloc[:,0]
ftse_daily

mcaps = {
    'SNAM': 15.54,
    'BEEWIZE': 0.00713,
    'FIDIA': 0.00923,
    'ALERION CLEAN POWER': 1.53,
    'ECOSUNTEK': 0.03367,
    'VIANINI INDR.': 0.03974,
    'LANDI RENZO': 0.12217,
    'RIZZOLI CRER.DLSM.GP.': 0.36429,
    "TOD'S": 1.41,
    'JUVENTUS FOOTBALL CLUB':  0.91242
}

for key in mcaps:
    mcaps[key] =  int(mcaps[key] * 10**9)

mcaps

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

S = risk_models.CovarianceShrinkage(selected_daily).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(ftse_daily, frequency=252, risk_free_rate=0.03)
delta

# plotting.plot_covariance(S, plot_correlation=True);

market_prior_daily = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=0.03)
market_prior_daily

market_prior_daily.plot.barh(figsize=(10,5));

P = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

Q = np.array([
    -0.044,
    0.268,
    0.263,
    0.126
]).reshape(-1, 1)

confidences = [
    0.33,
    0.33,
    0.33,
    0.67
]

bl = BlackLittermanModel(S, pi=market_prior_daily, P=P, Q=Q, omega="idzorek", view_confidences=confidences, risk_free_rate=0.03)
omega = bl.omega
np.diag(bl.omega)

"""### Posterior estimates"""

bl = BlackLittermanModel(S, pi='market', P=P, Q=Q, market_caps=mcaps, omega=omega, risk_aversion=delta, risk_free_rate=0.03)

# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl

rets_df = pd.DataFrame([market_prior_daily, ret_bl], index=["Prior", "Posterior"]).T
rets_df

rets_df.plot.bar(figsize=(12,8));

S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl);

"""### Portfolio Allocation"""

from pypfopt import EfficientFrontier, objective_functions

ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(-1,1))
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe(risk_free_rate=0)
weights = ef.clean_weights()
weights

plot_weights(weights, 'Daily - Max sharpe ratio - weight bounds (-1,1)')

ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(-1,1))
fig, ax = pyplot.subplots()
ef_max_sharpe = ef.deepcopy()
# plot_efficient_frontier(ef, ax=ax, show_assets=False)
# Find the tangency portfolio
ef_max_sharpe.max_sharpe(risk_free_rate=0)
ret_tangent, std_tangent, sharpe_ratio_tangent = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title(f"Efficient Frontier with Black-Litterman - Daily - weight bounds (-1,1)")
ax.legend()
pyplot.tight_layout()
pyplot.show()

ret_bl

weights.values()

print(f"Sharpe ratio:               {sharpe_ratio_tangent}")
print(f'Return portfolio - Daily    : {ret_tangent}')
print(f'Standard deviation - Daily  : {std_tangent}')
print(f'Kurtosis - Daily            : {ret_bl.kurtosis()}')
print(f'Skewness - Daily            : {ret_bl.skew()}')

"""## Monthly"""

selected_monthly.tail()

#market_prices
ftse_monthly = pd.read_excel(r'sample_data/Clean data.xlsx', sheet_name='FTSE Monthly ', index_col=0).iloc[:,0]
ftse_monthly

mcaps = {
    'SNAM': 15.54,
    'BEEWIZE': 0.00713,
    'FIDIA': 0.00923,
    'ALERION CLEAN POWER': 1.53,
    'ECOSUNTEK': 0.03367,
    'VIANINI INDR.': 0.03974,
    'LANDI RENZO': 0.12217,
    'RIZZOLI CRER.DLSM.GP.': 0.36429,
    "TOD'S": 1.41,
    'JUVENTUS FOOTBALL CLUB':  0.91242
}

for key in mcaps:
    mcaps[key] =  int(mcaps[key] * 10**9)

mcaps

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

S = risk_models.CovarianceShrinkage(selected_monthly, frequency=12).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(ftse_monthly, frequency=12, risk_free_rate=0.03)
delta

plotting.plot_covariance(S, plot_correlation=True);

market_prior_monthly = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=0.03)
market_prior_monthly

market_prior_monthly.plot.barh(figsize=(10,5));

bl = BlackLittermanModel(S, pi=market_prior_monthly, P=P, Q=Q, omega="idzorek", view_confidences=confidences, risk_free_rate=0.03)
omega = bl.omega
np.diag(bl.omega)

"""### Posterior estimates"""

bl = BlackLittermanModel(S, pi='market', P=P, Q=Q, market_caps=mcaps, omega=omega, risk_aversion=delta, risk_free_rate=0.03)

# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl

rets_df = pd.DataFrame([market_prior_monthly, ret_bl], index=["Prior", "Posterior"]).T
rets_df

rets_df.plot.bar(figsize=(12,8));

S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl);

"""### Portfolio allocation"""

from pypfopt import EfficientFrontier, objective_functions

ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(-1,1))
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe(risk_free_rate=0)
weights = ef.clean_weights()
weights

plot_weights(weights, 'Monthly - Max sharpe ratio - weight bounds (-1,1)')

ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(-1,1))
fig, ax = pyplot.subplots()
ef_max_sharpe = ef.deepcopy()
# plot_efficient_frontier(ef, ax=ax, show_assets=False)
# Find the tangency portfolio
ef_max_sharpe.max_sharpe(risk_free_rate=0)
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title(f"Efficient Frontier with Black-Litterman - Monthly - weight bounds (-1,1)")
ax.legend()
pyplot.tight_layout()
pyplot.show()

ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(-1,1))
ef_max_sharpe = ef.deepcopy()
ef_max_sharpe.max_sharpe(risk_free_rate=0)
ret_tangent, std_tangent, sharpe_ratio_tangent = ef_max_sharpe.portfolio_performance()

print(f"Sharpe ratio                  : {sharpe_ratio_tangent}")
print(f'Return portfolio - Monthly    : {ret_tangent}')
print(f'Standard deviation - Monthly  : {std_tangent}')
print(f'Kurtosis - Monthly            : {ret_bl.kurtosis()}')
print(f'Skewness - Monthly            : {ret_bl.skew()}')

"""# Bayesian Asset Allocation"""

portfolio_returns_daily = returns_daily.filter(selected_columns)

portfolio_returns_monthly = returns_monthly.filter(selected_columns)

import math
from numpy.linalg import inv

def bayesian_optimal_portfolio_daily(returns_df, l):
    mu_hat= (returns_df.mean()*252).to_numpy()
    sigma=returns_df.cov()*math.sqrt(252)
    T=returns_df.shape[0]
    mu_0=mu_hat+1*np.sqrt(np.diag(sigma))
    lambda_0=sigma*2

    mu_1 = inv(T * inv(sigma) + inv(lambda_0)) @ (T * inv(sigma) @ mu_hat + inv(lambda_0) @ mu_0)
    sigma_1 = inv(T * inv(sigma)*mu_hat + inv(lambda_0)*mu_0)

    w_bayes = (1 / l) * inv(sigma_1) @ mu_1
    w_bayes = w_bayes * 0.75 / sum(w_bayes)
    return w_bayes


def bayesian_optimal_portfolio_monthly(returns_df, l):
    mu_hat= (returns_df.mean()*12).to_numpy()
    sigma=returns_df.cov()*math.sqrt(12)
    T=returns_df.shape[0]
    mu_0=mu_hat+1*np.sqrt(np.diag(sigma))
    lambda_0=sigma*2

    mu_1 = inv(T * inv(sigma) + inv(lambda_0)) @ (T * inv(sigma) @ mu_hat + inv(lambda_0) @ mu_0)
    sigma_1 = inv(T * inv(sigma)*mu_hat + inv(lambda_0)*mu_0)

    w_bayes = (1 / l) * inv(sigma_1) @ mu_1
    w_bayes = w_bayes * 0.75 / sum(w_bayes)
    return w_bayes

bayesian_weights_daily = bayesian_optimal_portfolio_daily(returns_df=portfolio_returns_daily, l=0.9)
bayesian_weights_monthly = bayesian_optimal_portfolio_monthly(returns_df=portfolio_returns_monthly, l=0.9)

plot_weights(pd.Series(bayesian_weights_daily, index=portfolio_returns_daily.columns).to_dict(), 'Daily ')

risk_free_rate = 0.22
risk_free_rate_monthly = 0.16

plot_weights(pd.Series(bayesian_weights_monthly, index=portfolio_returns_monthly.columns).to_dict(), 'Monthly ')

from scipy.stats import kurtosis, skew
# Daily
mu_hat = (portfolio_returns_daily.mean()*252).to_numpy()
sigma = portfolio_returns_daily.cov()*math.sqrt(252)
ef = EfficientFrontier(mu_hat, sigma, weight_bounds=(-1,1))
ef_max_sharpe = ef.deepcopy()
ef_max_sharpe.max_sharpe(risk_free_rate=0)
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()

print(f"Sharpe ratio: {(ret_tangent-risk_free_rate)/std_tangent}")
print(f'Return portfolio - Daily    : {ret_tangent}')
print(f'Standard deviation - Daily  : {std_tangent}')
print(f'Kurtosis - Daily            : {kurtosis(mu_hat)}')
print(f'Skewness - Daily            : {skew(mu_hat)}')

from scipy.stats import kurtosis, skew
# Monthly
mu_hat = (portfolio_returns_monthly.mean()*12).to_numpy()
sigma = portfolio_returns_monthly.cov()*math.sqrt(12)
ef = EfficientFrontier(mu_hat, sigma, weight_bounds=(-1,1))
ef_max_sharpe = ef.deepcopy()
ef_max_sharpe.max_sharpe(risk_free_rate=0)
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
print(f"Sharpe ratio: {(ret_tangent-risk_free_rate_monthly)/std_tangent}")
print(f'Return portfolio - Monthly    : {ret_tangent}')
print(f'Standard deviation - Monthly  : {std_tangent}')
print(f'Kurtosis - Monthly            : {kurtosis(mu_hat)}')
print(f'Skewness - Monthly            : {skew(mu_hat)}')
import matplotlib.pyplot as plt
import pandas as pd


def get_data():
    # Load actuals DA and ID prices
    data = pd.read_csv('finalData_hourly_upd.csv')[['DateTime', 'vwap', 'dayAhead_NL']]
    
    # Prepare and merge prediction dfs
    pred_dfs = {
        'narrow': pd.read_csv('predictions/bnn_pred_narrow.csv'),
        'medium': pd.read_csv('predictions/bnn_pred_medium.csv'),
        'wide': pd.read_csv('predictions/bnn_pred_wide.csv'),
        'nn': pd.read_csv('predictions/bnn_pred_nn.csv')
    }
    
    # Merge prediction dfs to main data on DateTime (label variables)
    for key, df in pred_dfs.items():
        df = df[['DateTime', 'bnn', 'ci_lower', 'ci_upper']].rename(
            columns={
                'bnn': f'bnn_{key}',
                'ci_lower': f'ci_lower_{key}',
                'ci_upper': f'ci_upper_{key}'
            }
        )
        data = pd.merge(data, df, on='DateTime', how='inner')

    return data


def determine_action(row, model_key, trade_cost):
    '''
    Helper method determines Buy/Sell/Hold action for a given prior model
    '''
    ci_lower_key = f'ci_lower_{model_key}'
    ci_upper_key = f'ci_upper_{model_key}'
    action = 'Hold'
    correct = False
    
    # DA exceeds upper bound => short the DA price 
    # (correct checks if action is profitable)
    if row['dayAhead_NL'] > row[ci_upper_key]:
        action = 'Sell'
        correct = row['dayAhead_NL'] > row['vwap'] + trade_cost

    # DA exceeds lower bound => go long the DA price
    elif row['dayAhead_NL'] < row[ci_lower_key]:
        action = 'Buy'
        correct = row['dayAhead_NL']  < row['vwap'] - trade_cost
    
    return {
        'DateTime': row['DateTime'],
        'Action': action,
        'Correct': correct,
        'dayAhead_NL': row['dayAhead_NL'],
        'vwap': row['vwap'],
        ci_lower_key: row[ci_lower_key],
        ci_upper_key: row[ci_upper_key]
    }


def analyse_strategy(df, model_key, unit_size, trade_cost):
    '''
    Method calculates the profit function defined on page 7.
    '''
    actions = []
    datetime_points = []
    profits = []
    cumulative_profits = []
    correctness = []
    cumulative_profit = 0
    
    for index, row in df.iterrows():
        action_data = determine_action(row, model_key, trade_cost)
        action = action_data['Action']
        correct = action_data['Correct']
        profit = 0
        
        if action == 'Sell':
            profit = row['dayAhead_NL'] - (row['vwap'] + trade_cost)
        elif action == 'Buy':
            profit = row['vwap'] - (row['dayAhead_NL'] + trade_cost)
        
        cumulative_profit += unit_size * profit
        
        # save results if buy or sell action initiated
        if action in ['Buy', 'Sell']:
            actions.append(action)
            datetime_points.append(action_data['DateTime'])
            profits.append(profit)
            cumulative_profits.append(cumulative_profit)
            correctness.append(correct)
    
    result_df = pd.DataFrame({
        'DateTime': datetime_points,
        'Action': actions,
        'Correct': correctness,
        'Profit': profits,
        'Cumulative Profit': cumulative_profits
    })

    return result_df


def results_statistics(df):

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Calc nr of Buy/Sell actions
    num_sell = df[df['Action'] == 'Sell'].shape[0]
    num_buy = df[df['Action'] == 'Buy'].shape[0]
    
    # Calc fraction of profitable trades
    fraction_correct = df['Correct'].mean()
    
    # Calc avg profitability
    average_profitability = df['Profit'].mean()
    
    # Find time intervals with most trades
    df['Hour'] = df['DateTime'].dt.hour
    most_entries_time = df['Hour'].mode()[0]
    
    summary_stats = {
        "Number of Sell entries": num_sell,
        "Number of Buy entries": num_buy,
        "Fraction of correct entries": fraction_correct,
        "Average profitability": average_profitability,
        "Hour with most entries": most_entries_time
    }
    
    return summary_stats


def plot_confidence_bounds(df, model_key, trade_cost, start_date=None, end_date=None):

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Filter data or use whole set if no range provided
    if start_date and end_date:
        df_filtered = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]
    else:
        df_filtered = df
    
    ci_lower_key = f'ci_lower_{model_key}'
    ci_upper_key = f'ci_upper_{model_key}'
    
    plt.figure(figsize=(14, 7))   

    # Plot DA and ID prices
    plt.plot(df_filtered['DateTime'], df_filtered['dayAhead_NL'], label='Day Ahead Price (DA)', color='blue')
    plt.plot(df_filtered['DateTime'], df_filtered['vwap'], label='Intraday Price (ID)', color='orange')

    # Plot confInt
    plt.fill_between(df_filtered['DateTime'], df_filtered[ci_lower_key], 
                     df_filtered[ci_upper_key], color='gray', alpha=0.3, label='95% Conf Int.')
    
    # Mark points where DA exceeds Lower or Upper
    buys = df_filtered[df_filtered.apply(lambda row: determine_action(row, model_key, trade_cost)['Action'] == 'Buy', axis=1)]
    sells = df_filtered[df_filtered.apply(lambda row: determine_action(row, model_key, trade_cost)['Action'] == 'Sell', axis=1)]
    plt.scatter(buys['DateTime'], buys['dayAhead_NL'], color='green', label='Buy (DA < L)', marker='o')
    plt.scatter(sells['DateTime'], sells['dayAhead_NL'], color='red', label='Sell (DA > U)', marker='x')
    plt.xlabel('Date Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Confidence Bounds and Prices' + (f' from {start_date} to {end_date}' if start_date and end_date else ''), fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def main():

    full_data = get_data()
    start_date = '2023-07-05'          # allow for stabilisation of wide prior
    full_data['DateTime'] = pd.to_datetime(full_data['DateTime'])
    start_date = pd.to_datetime(start_date)
    combined_data = full_data[full_data['DateTime'] >= start_date]

    unit_size = 1                      # Define unit size in MWh
    trade_cost = 0.75                  # transaction cost for a single trade in EUR

    results_nn = analyse_strategy(combined_data, 'nn', unit_size, trade_cost)
    results_narrow = analyse_strategy(combined_data, 'narrow', unit_size, trade_cost)
    results_medium = analyse_strategy(combined_data, 'medium', unit_size, trade_cost)
    results_wide = analyse_strategy(combined_data, 'wide', unit_size, trade_cost)

    stats_nn = results_statistics(results_nn)
    print(stats_nn)
    # stats_narrow = results_statistics(results_narrow)
    # print(stats_narrow)
    # stats_medium = results_statistics(results_medium)
    # print(stats_medium)
    # stats_wide = results_statistics(results_wide)
    # print(stats_wide)

    plot_confidence_bounds(full_data, 'nn', trade_cost, '2023-07-05', '2023-08-05')
    # plot_confidence_bounds(full_data, 'nn', trade_cost) # full set

    # list all actions
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_nn)
        # print(40*'-')
        # print(results_narrow)
        # print(40*'-')
        # print(results_medium)
        # print(40*'-')
        # print(results_wide)


if __name__=='__main__':
    main()

import matplotlib.pyplot as plt
import pandas as pd


def plot_avg_hourly_losses(loss, start_date=None):
    '''
    Method calculates and plots the average hourly loss 
    from start_date to the end of the dataset for 
    either MAE or sMAPE losses. Additionally prints the 
    avg losses for each model from start_date to end of dataset
    '''
    # load csv's
    narrow_df = pd.read_csv('losses/bnn_loss_narrow.csv')
    medium_df = pd.read_csv('losses/bnn_loss_medium.csv')
    wide_df = pd.read_csv('losses/bnn_loss_wide.csv')
    nn_df = pd.read_csv('losses/bnn_loss_nn.csv')

    # convert the DateTime col to datetime object
    nn_df['DateTime'] = pd.to_datetime(nn_df['DateTime'])
    narrow_df['DateTime'] = pd.to_datetime(narrow_df['DateTime'])
    medium_df['DateTime'] = pd.to_datetime(medium_df['DateTime'])
    wide_df['DateTime'] = pd.to_datetime(wide_df['DateTime'])

    if start_date:             # filter if start date is provided
        start_date = pd.to_datetime(start_date)
        nn_df = nn_df[nn_df['DateTime'] >= start_date]
        narrow_df = narrow_df[narrow_df['DateTime'] >= start_date]
        medium_df = medium_df[medium_df['DateTime'] >= start_date]
        wide_df = wide_df[wide_df['DateTime'] >= start_date]
        
    # extract hour from each datetime
    nn_df['Hour'] = nn_df['DateTime'].dt.hour
    narrow_df['Hour'] = narrow_df['DateTime'].dt.hour
    medium_df['Hour'] = medium_df['DateTime'].dt.hour
    wide_df['Hour'] = wide_df['DateTime'].dt.hour

    # group by hour and calculate avg loss
    if loss == 'MAE': 
        nn_hourly_avg = nn_df.groupby('Hour')['MAE'].mean()
        narrow_hourly_avg = narrow_df.groupby('Hour')['MAE'].mean()
        medium_hourly_avg = medium_df.groupby('Hour')['MAE'].mean()
        wide_hourly_avg = wide_df.groupby('Hour')['MAE'].mean()

    elif loss == 'sMAPE':
        nn_hourly_avg = nn_df.groupby('Hour')['sMAPE'].mean()
        narrow_hourly_avg = narrow_df.groupby('Hour')['sMAPE'].mean()
        medium_hourly_avg = medium_df.groupby('Hour')['sMAPE'].mean()
        wide_hourly_avg = wide_df.groupby('Hour')['sMAPE'].mean()

    # plot results
    plt.figure(figsize=(12, 6))
    plt.plot(wide_hourly_avg.index, wide_hourly_avg.values, label='Wide Prior', marker='.')
    plt.plot(medium_hourly_avg.index, medium_hourly_avg.values, label='Medium Prior', marker='.')
    plt.plot(narrow_hourly_avg.index, narrow_hourly_avg.values, label='Narrow Prior', marker='.')
    plt.plot(nn_hourly_avg.index, nn_hourly_avg.values, label='Narrowest Prior', marker='.')

    plt.title(f'Average Loss ({loss}) at Each Hour of the Day', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel(f'Average {loss}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(range(24), fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    nn_avg_loss = nn_df[loss].mean()
    narrow_avg_loss = narrow_df[loss].mean()
    medium_avg_loss = medium_df[loss].mean()
    wide_avg_loss = wide_df[loss].mean()

    print(f"Average {loss} for Narrowest Prior: {nn_avg_loss:.2f}")
    print(f"Average {loss} for Narrow Prior: {narrow_avg_loss:.2f}")
    print(f"Average {loss} for Medium Prior: {medium_avg_loss:.2f}")
    print(f"Average {loss} for Wide Prior: {wide_avg_loss:.2f}")
    

def plot_all_predictions(start_date=None, end_date=None):
    '''
    Method plots the actuals, model predictions, and confidence bounds
    for each of the prior models given a start and end date 
    (between 04-05-2023 - 31-12-2023).
    '''
    # load prediction csv's
    nn_predictions_df = pd.read_csv('predictions/bnn_pred_nn.csv')
    narrow_predictions_df = pd.read_csv('predictions/bnn_pred_narrow.csv')
    medium_predictions_df = pd.read_csv('predictions/bnn_pred_medium.csv')
    wide_predictions_df = pd.read_csv('predictions/bnn_pred_wide.csv')
    
    # Convert DateTime cols to datetime object
    nn_predictions_df['DateTime'] = pd.to_datetime(nn_predictions_df['DateTime'])
    narrow_predictions_df['DateTime'] = pd.to_datetime(narrow_predictions_df['DateTime'])
    medium_predictions_df['DateTime'] = pd.to_datetime(medium_predictions_df['DateTime'])
    wide_predictions_df['DateTime'] = pd.to_datetime(wide_predictions_df['DateTime'])
    

    def plot_predictions(predictions_df, title):
        '''
        Helper function creates the plot for a given prediction df
        '''
        if start_date and end_date:         # filter df given start and end date
            mask = (predictions_df['DateTime'] >= start_date) & (predictions_df['DateTime'] <= end_date)
            predictions_df = predictions_df.loc[mask]
        
        # create sequenece of date indices to prevent overcrowding in plot
        x_indices = range(0, len(predictions_df['DateTime']), max(1, len(predictions_df['DateTime']) // 10))
        x_labels = predictions_df['DateTime'].iloc[x_indices]
        x_labels = pd.to_datetime(x_labels)

        plt.figure(figsize=(12, 6))
        plt.plot(predictions_df['DateTime'], predictions_df['actual'], label='Actual', alpha=0.75, color='blue')
        plt.plot(predictions_df['DateTime'], predictions_df['bnn'], label='Predicted', alpha=0.75, color='orange')
        plt.fill_between(predictions_df['DateTime'], 
                        predictions_df['ci_lower'], 
                        predictions_df['ci_upper'], color='gray', alpha=0.2, label='95% CI')
        plt.grid(True)
        plt.title(title, fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('VWAP', fontsize=14)
        plt.xticks(ticks=x_labels, labels=x_labels, rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim((-600, 800))
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


    # Plot prediction dfs
    plot_predictions(nn_predictions_df, 'BNN Narrowest Prior')
    plot_predictions(narrow_predictions_df, 'BNN Narrow Prior')
    plot_predictions(medium_predictions_df, 'BNN Medium Prior')
    plot_predictions(wide_predictions_df, 'BNN Wide Prior')
    

def main():
    # plot_avg_hourly_losses('sMAPE')
    plot_avg_hourly_losses('sMAPE', '2023-07-05')
    # plot_all_predictions(start_date='2023-05-04', end_date='2023-08-31')
    # plot_all_predictions(start_date='2023-07-05', end_date='2023-12-31')
    # plot_all_predictions()


if __name__=='__main__':
    main()

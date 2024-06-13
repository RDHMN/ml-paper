import pandas as pd
import numpy as np
import pickle, random
import torch
import torch.optim as optim
import torchbnn as bnn
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# set seed(s) for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# one-hot encoded time dummies 
# is_holiday: time dummy for Dutch public holidays
time_columns = ['day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3',
                'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
                'hour_of_day_0', 'hour_of_day_1', 'hour_of_day_2', 'hour_of_day_3',
                'hour_of_day_4', 'hour_of_day_5', 'hour_of_day_6', 'hour_of_day_7',
                'hour_of_day_8', 'hour_of_day_9', 'hour_of_day_10', 'hour_of_day_11',
                'hour_of_day_12', 'hour_of_day_13', 'hour_of_day_14', 'hour_of_day_15',
                'hour_of_day_16', 'hour_of_day_17', 'hour_of_day_18', 'hour_of_day_19',
                'hour_of_day_20', 'hour_of_day_21', 'hour_of_day_22', 'hour_of_day_23', 'is_holiday']


def get_data():
    data = 'finalData_hourly_upd.csv'
    columns = ['DateTime', 'vwap', 'solar_fc_da', 'coal_fc_da', 
                'gas_fc_da', 'demand_fc_da', 'dayAhead_fc_NL', 
                'wind_fc_da', 'temp_fc_da', 'gas_price']

    data_df = pd.read_csv(data, usecols=columns + time_columns)
        
    return data_df


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    '''
    sMAPE formula implementation (page 7)
    '''
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2
    return np.mean(numerator / denominator) * 100


def build_bnn_model(input_dim, hidden_layers, 
                    neurons_per_layer, activation_function, 
                    dropout, prior_mu, prior_sigma):
    '''
    Method creating the BNN model, allows for a wide array of hyperparameter combinations.
    The structure of the BNN is explained in the Model Setup section.
    '''
    layers = []
    for _ in range(hidden_layers):
        layers.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, 
                                      in_features=input_dim, out_features=neurons_per_layer))
        layers.append(getattr(nn, activation_function)())
        layers.append(nn.Dropout(dropout))
        input_dim = neurons_per_layer
    
    layers.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, 
                                  in_features=input_dim, out_features=1))
    return nn.Sequential(*layers)


def train_bnn_rolling_window(df, target, learning_rate, hidden_layers, 
                             neurons_per_layer, epochs, batch_size, 
                             activation_function, dropout, prior_mu, 
                             prior_sigma, patience):
    '''
    Method implements and trains the BNN on a rolling window,
    saves the model predictions, loss history, and models to csv files.
    Returns a dictionary of average losses, predictions, and the exact MAE/sMAPE loss history
    (code for validation set is commented out, no validation provided better results)
    '''
    ts_data = df[target]
    time_columns = [col for col in df if col.startswith('day_of_week_') 
                    or col.startswith('hour_of_day_') or col == 'is_holiday']
    feature_columns = [col for col in df.columns if col not in time_columns + [target, 'DateTime']]

    training_segment = int(4 * 30.5 * 24)                           # 4 months of training data
    validation_test_segment = int(24)                               # 1 day
    smape_scores, mae_scores, rmse_scores = [], [], []
    results = []
    loss_history = []

    input_dim = len(feature_columns) + len(time_columns)
    model = build_bnn_model(input_dim, hidden_layers, 
                            neurons_per_layer, activation_function, 
                            dropout, prior_mu, prior_sigma)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()                                         # MAE loss

    for start_index in range(0, len(df) - training_segment - validation_test_segment, validation_test_segment):
        train_end = start_index + training_segment
        # val_start = train_end
        # val_end = val_start + validation_test_segment 
        # test_start = val_end
        # test_end = test_start + validation_test_segment 
        test_start = train_end
        test_end = test_start + validation_test_segment 

        if test_end >= len(df):
            break

        print(f"Training window from index {start_index} to {train_end}")
        X_train = df.iloc[start_index:train_end]
        y_train = ts_data.iloc[start_index:train_end].values 
        # X_val = df.iloc[val_start:val_end]
        # y_val = ts_data.iloc[val_start:val_end].values
        X_test = df.iloc[test_start:test_end]
        y_test = ts_data.iloc[test_start:test_end].values

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # scale feature columns
        X_train_scaled = scaler_X.fit_transform(X_train[feature_columns])
        # X_val_scaled = scaler_X.transform(X_val[feature_columns])
        X_test_scaled = scaler_X.transform(X_test[feature_columns])

        # append time dummies
        X_train_final = np.hstack([X_train_scaled, X_train[time_columns].values])
        # X_val_final = np.hstack([X_val_scaled, X_val[time_columns].values])
        X_test_final = np.hstack([X_test_scaled, X_test[time_columns].values])

        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        # y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        # create TensorDatasets
        train_dataset = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), 
                                      torch.tensor(y_train_scaled, dtype=torch.float32))
        # val_dataset = TensorDataset(torch.tensor(X_val_final, dtype=torch.float32), 
                                    # torch.tensor(y_val_scaled, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), 
                                    torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # best_val_loss = float('inf')
        # epochs_no_improve = 0

        # start model training for current window
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.flatten(), y_batch)
                loss.backward()
                optimizer.step()
            
            # model.eval()
            # val_loss = 0
            # with torch.no_grad():
            #     for X_batch, y_batch in val_loader:
            #         outputs = model(X_batch)
            #         val_loss += criterion(outputs.flatten(), y_batch).item()
            
            # val_loss /= len(val_loader)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} completed with training loss: {loss.item()}")
                    #   , validation loss: {val_loss}")

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_no_improve = 0
            #     best_model_state = model.state_dict()
            # else:
            #     epochs_no_improve += 1
            
            # if epochs_no_improve >= patience:
            #     print(f"Early stopping triggered after {epoch + 1} epochs")
            #     break

        # model.load_state_dict(best_model_state)

        # start model evaluation for current window 
        model.eval()
        y_preds = []
        y_std_devs = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                preds = model(X_batch)
                y_preds.append(preds)
                y_std_devs.append(preds)

        y_preds = torch.cat(y_preds).numpy().flatten()
        y_std_devs = torch.cat(y_std_devs).numpy().flatten()
        
        y_preds = scaler_y.inverse_transform(y_preds.reshape(-1, 1)).flatten()
        y_std_devs = scaler_y.inverse_transform(y_std_devs.reshape(-1, 1)).flatten()
        
        # save actuals, model predictions (mean), and confidence intervals
        results.extend(zip(X_test['DateTime'], y_test, y_preds, 
                           y_preds - 1.96 * y_std_devs, y_preds + 1.96 * y_std_devs))

        # keep track of losses per window
        smape = symmetric_mean_absolute_percentage_error(y_test, y_preds)
        mae = mean_absolute_error(y_test, y_preds)
        rmse = np.sqrt(mean_squared_error(y_test, y_preds))
        smape_scores.append(smape)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # keep track of individual losses
        mae_per_prediction = np.abs(y_test - y_preds)
        smape_per_prediction = 200 * np.abs(y_test - y_preds) / (np.abs(y_test) + np.abs(y_preds))
        loss_history.extend(zip(X_test['DateTime'], mae_per_prediction, smape_per_prediction))

    results_df = pd.DataFrame(results, columns=['DateTime', 'actual', 'bnn', 'ci_lower', 'ci_upper'])
    results_df.to_csv('bnn_pred_nn.csv', index=False)

    loss_history_df = pd.DataFrame(loss_history, columns=['DateTime', 'MAE', 'sMAPE'])
    loss_history_df.to_csv('bnn_loss_nn.csv', index=False)

    # save the trained BNN model
    with open('bnn_nn.pkl', 'wb') as file:
        pickle.dump(model, file)

    return {
        'avg_smape': np.mean(smape_scores),
        'avg_mae': np.mean(mae_scores),
        'avg_rmse': np.mean(rmse_scores),
        'predictions': results_df,
        'loss_history': loss_history_df
    }


def main():
    '''
    Main method to set hyperparameters, train the BNN, 
    and report average losses of the model
    '''
    learning_rate = 0.0005
    hidden_layers = 3
    neurons_per_layer = 56
    epochs = 50
    batch_size = 100
    activation_function = 'ELU'
    dropout = 0.1
    prior_mu = 0
    prior_sigma = 0.00005
    patience = 20  # no early stopping seems to have best performance

    df = get_data() 
    results = train_bnn_rolling_window(df, 'vwap', learning_rate, hidden_layers,
                                       neurons_per_layer, epochs, batch_size,
                                       activation_function, dropout, prior_mu,
                                       prior_sigma, patience)

    print('MAE: ', results['avg_mae'])
    print(40 * '-')
    print('RMSE: ', results['avg_rmse'])
    print(40 * '-')
    print('sMAPE: ', results['avg_smape'])


if __name__ == '__main__':
    main()

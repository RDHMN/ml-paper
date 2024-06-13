# ml-paper

This repository contains all code for the Machine Learning (EBC4257) final paper excluding data.

bnn_model.py: \
Rolling window implementation of the BNN for the varying priors.
\
\
evaluate_strategy.py: \
Code for evaluating the proposed arbitrage strategy using the prediction confidence bounds.
\
\
plots.py: \
Code for several plots used in the paper.
\
\
Additionally, the losses of each model are provided in the losses folder.
\
————————————————————————————————————————\
The data used for the analysis originates from 2 main sources:\
\
EnAppSys\
Link: https://www.enappsys.com/market-data-platform/

Data:  
1. D-1 solar generation forecast
2. D-1 coal generation forecast
3. D-1 gas generation forecast	
4. D-1 demand volume forecast
5. H-1 day-ahead price forecast and actual prices
6. D-1 wind generation forecast
7. D-1 temperature forecast
8. Actual gas price (determined daily)


NordPool\
Link: https://developers.nordpoolgroup.com/reference/welcome


Data: 	\
Information on intraday market trades for the NL bidding zone
Trades are filtered between 10 to 1 hour before delivery to mitigate speculative effects.Filtered trades are aggregated per delivery hour, we use the volume weighted average as dependent variable for the analysis.

Additionally, we include hour of day, day of week, and Dutch public holiday time dummies.

————————————————————

The actual dataset is omitted from the repository as data from both sources isn’t freely available.

The preprocessing steps are as follows:

1. Retrieve data from either the EnAppSys or Nordpool API.
2. For the EnAppSys variables, we check for missing entries and replace them with their corresponding H-1 forecasts of the same source.
3. Nordpool trades are filtered and aggregated as specified above and on page 4.
4. Merge the Nordpool and EnAppSys data on datetime and delete all rows missing from the Nordpool data.


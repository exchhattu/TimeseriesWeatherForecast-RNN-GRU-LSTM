# Weather forecast using recurrent neural network

## **Motivation**
* Analyze time series data. Here, weather forecasting data was used. However 
  with minimal modification, __the program can be used in the time series data from 
  different domains such as finance or health care__.
* The goal is to predict temperature of the next 12 or 24 hours
  as time series data for weather forecasting was tested. 
* Compare performance using mean square error and mean absolute error 

## **Requirements** 
* Python (3.6.0)
* Pandas (0.24.1)
* Numpy (1.16.0)
* Keras (2.2.4) 
* Tensorflow (1.13.1)
* Jupyter (4.4.0)
* Matplotlib (3.0.2) and Seaborn (0.9.0)

## **Directory Structure**
- __src__: contains scripts for library and binary
- __notebooks__: jupyter notebooks of optimized models 
- __DLWP__: jupyter notebooks of the examples provided in the book and other
  comparison tests. 

## **Usage** 

* Long Short Term Memory 
```
python3 ./src/LSTM.py 
```

* Gated Recurrent Unit 
```
python3 ./src/GRU.py 
```

* Example
```
Run GRU layer with embedded feature

python3 ./src/GRU.py -e -i path/to/input  

Options are same for both programs.

-i: path to input file.
-p: hour for prediction (default:24) 
-e: incorporate embedded features (default: False)
-d: incorporate encoded features (default: False)  
```

## **Updated Summary**

* First, model with gated recurrent unit (GRU) was implemented and tested. 
 The mean absolute error (MAE) of [0.30](https://bit.ly/2kqrO4K) was the lowest while generating 
 a model with the dropout scheme. This error was yielded at 30 epoch on the validation dataset.
 Afterward, both functions are converged. The model showed similar performance on the test dataset.
 When assessment was changed from [MAE to MSE](https://bit.ly/2kfketZ) (mean square error), 
 the best performance was 0.13 MSE (0.36 RMSE and almost equal to MAE) on validation and test dataset in the same scheme. 
 However, the model was converged at 22 epoch or earlier. This indicates there
 is no effect of outlier and therfore, MSE is used for further analysis since it is easy to compute due to its L2 norm.

* Similarly, long short-term memory(LSTM) was also tested by replacing GRU with [MAE](https://bit.ly/2lUMEd8) 
  and [MSE](https://bit.ly/2jTdaTq) respectively. However, the performance slight decreased and 
  was differed compared with its GRU counterpart. But there is no change 
  for selection of loss functions. Thus, further analysis is only with MSE.

* Add-in:
 I further explored to optimize a model to avoid overfitting in which MSE was used for all cases. 
 I mainly focus on the few things, which are listed below:
   1. Data preparation
     Around half a million data was included in the experiment. These data were divided into train, 
     validation, and test group in 8:1:1 ratio.

   2. Parameter selection
     The number of offset data (prior) taken from the current time was changed. 
     Similarly, the step size introduced to select the data was also modified.

   3. Incorporate temporal data
     Temporal information was excluded in the example shown in the book. 
     However, this information was incorporated in two different ways. 
     First, day, month, and time were converted into numerical values in such a way that January and December 
     are close to each other. Second, using an embedded layer from Keras, four columns - year, month, day, 
     and time were embedded and concatenated with numeric attributes.

## **Methods and Performances**
  - Configuration: 20 epochs with 200 steps/epoch, 0.03 dropouts and recurrent dropouts
  - Feature Definition:
    * Default feature indicates 14 different attributes that have numerical values.
    * -e in the program indicates embedded features that were combined with 14 attributes.
     They are Day, month, year, and time.
    * -d in the program refers to the inclusion of encoded day, month, and time with 14 attributes. 
      For example, a month is encoded in such a way that January and December are very close to each other.
  - Among the tested models, the stacking layer with the default and embedded features for GRU and LSTM were selected.
  - Temperature prediction of [24 hours](https://github.com/exchhattu/TimeseriesWeatherForecast-RNN-GRU-LSTM/blob/master/notebooks/OptimizedModel_default.ipynb) 
    * Four models, two for GRU and two for LSTM, with the default and embedded features, 
      yielded similar mean square error (MSE) of 0.12 on the test dataset. 
      Loss was plotted as function of the number of epoch for GRU and LSTM with the default and embedded features.

  - Models for forecasting temperature of next [12 hours](https://github.com/exchhattu/TimeseriesWeatherForecast-RNN-GRU-LSTM/blob/master/notebooks/OptimizedModel_12hrs.ipynb) 
    * Using similar configuration, the model was generated to predict 12 hours' weather using similar 
      parameters. The model that included embedded features yielded a better performance of 0.09 
      mean squared error. This is slightly improved over the same model generated for 24 hours'
     prediction. However, there is no performance improvement (0.12 MSE) for 12 hour's weather prediction 
     over 24 hour's when only the default feature with a similar configuration was used.

    * Similar experiments were also carried out to generated models using GRU. Two models yielded 0.09 
      and 0.10 mean square error with embedded and default features respectively.

## **Future Direction**
  - The behavior of time series data is highly stochastic, which increases the uncertainty in finding 
    a pattern in the data. Nonetheless, recent advancements in machine learning algorithm such as deep 
    learning can improve the prediction performance with confidence.
  - I am interested to explore multiple parameters in order to optimize the model 
    to yield better performance in the availability of computing resources. Moreover, my idea is to 
    expand the utility of a program to employ the similar time-series data from different domains.

## Disclamier
Opinions expressed are solely my own and do not express the views or opinions of my employer. 
The author assumes no responsibility or liability for any errors or omissions in the content of this site. 
The information contained in this site is provided on an “as is” basis with no guarantees of completeness, 
accuracy, usefulness or timeliness.

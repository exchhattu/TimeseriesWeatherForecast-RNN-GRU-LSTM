# Weather forecast using recurrent neural network

## **Motivation**
* Analyze time series data. Here, weather forecasting data was used. However 
  with minimal modification, __the program can be used in the time series data from 
  different domains such as finance or health care__.
* The goal is to predict temperature of the next 12 or 24 hours
  as time series data for weather forecasting was tested. 
* Compare mean square error and mean absolute error as loss function

## **Requirements** 
* Python (3.6.0)
* Pandas ()
* Numpy (1.16.0)
* Keras (2.2.4) 
* Tensorflow (1.13.1)
* Juypter ()
* Matplotlib and Seaborn

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
* An example provided for gated recurrent unit (GRU) in [Deep Learning with Python](https://bit.ly/346tOkH)
  was repeated. The mean absolute error (MAE) of [0.30](https://bit.ly/2kqrO4K) was the lowest while 
  generating a model with dropout scheme. This error was yielded at 30 epoch on validation dataset. 
  Afterward, both functions are converged. The model showed similar performance on test dataset. 
  When loss function was changed from [MAE to MSE](https://bit.ly/2kfketZ) (mean square error), the best performance was 
  0.13 MSE on validation and test dataset in same scheme. However, model was converged earlier at 22 epoch.  Although 
  both metrics can recaptiulate the acutal data with score of 0.00, they grow differently. 

* Long short-term memory(LSTM) was also tested by replacing GRU
  with [MAE](https://bit.ly/2lUMEd8) and [MSE](https://bit.ly/2jTdaTq) respectively. 
  However, the performance slight decreased and the trend of loss functions 
  were differed compared with its GRU counterpart.

* Add-in: 
  I further explored to optimize a model to avoid overfitting in which MSE was 
  used as loss function for all cases. I mainly focus on the few things, which 
  are listed below:

   1. Data preparation
      Around half million data was included for the experiment. These data were
      divided into train, validation, and test group in 8:1:1 ratio.

   2. Parameter selection
      The number of offset data (prior) taken from current time was
      changed. Similarly, step size introduced to select the data was also 
      modified. 
      
   3. Incorporate temporal data
      Temporal information was excluded in the example shown in the book. However,
      this information was incorporated in two different ways. First, day,
      month, and time were converted into numerical values in such a way that January
      and December are close to each other. Second, using embedded layer from
      Keras, four columns - year, month, day, and time were embedded and
      concatenated with numeric attributes.

## **Methods and Performances**
   - Configuration: 20 epochs with 200 steps/epoch with 0.03 dropouts and recurrent dropouts
   - Feature Definition:
      * Default feature indicates 14 different attributes that have numerical values.
      * -e in program indicates embedded features that were combined with 14 attributes.
        They are Day, month, year, and time.
      * -d in program refers the inclusion of encoded day, month, and time with 14
        attributes. For example, month is encoded in such a way that January and 
        December are very close to each other.
   - Among the tested models, stacking with default and embedded
     features for GRU and LSTM were selected. 

   - 24 hours (default) weather prediction summary
      * Four models, two for GRU and two for LSTM, with default and embedded features yielded 
        similar mean square error (MSE) of 0.12 on test dataset. [Loss function of epoch
        for GRU and LSTM with default and embedded features](https://bit.ly/2zuTkSD) were plotted. 

   - [Models](https://bit.ly/30LqDgj) for 12 hours prediction
      * Using similar configuration, the model was generated to predict 12 hours' weather using similar
        parameters. The model that included embedded features yielded better performance of 
        0.09 mean squared error. This is slightly improved over the same model generated for 24 hours' 
        prediction. However, there is no performance improvement (0.12 MSE) for 12 hour's weather prediction 
        over 24 hour's when only default feature with similar configuration was used.

      * Similar experiments were also carried out to generated model using GRU.
        Two models yielded 0.09 and 0.10 mean square error with embedded and
        default features respectively.

## **Future Direction**
  - The behavior of time series data is highly stocastic, which increases the
    uncertanity in finding a pattern in the data. Nonetheless, recent
    advancement in machine learning algorithm such as deep learning can improve 
    the prediction performance with confidence.
  - I am interested to explore multiple parameters and different models in order to
    optimize the model to yield better performance in the availability of
    computing resources. Moreover, my idea is to expand the utility of a program to 
    employ in the similar time series data from different domains.

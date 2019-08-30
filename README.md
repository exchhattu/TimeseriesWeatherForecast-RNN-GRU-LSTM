# Weather forecast using recurrent neural network

## **Motivation**
* Analyze time series data. Here, weather forecasting data was used. However 
  with minimum modification the program can be used in the similar time
  series data from different domains such as finance, and health care.
* Here, since time series data was used, the goal is to predict temperature of the next 12 or 24 hours 

## **Requirements** 
* Python (3.6.0)
* Pandas ()
* numpy (1.16.0)
* keras (2.2.4) 
* Tensorflow (1.13.1)
* Juypter ()
* Matplotlib and Seaborn

## **Directory Structure**
- *src*: contains library and binary scripts. 
- *notebook*: jupyter notebook for logs of optimized models 
- *DLWP*: jupyter notebook of examples provided in the book 

## **Usage** 

* LSTM layer
```
python3 ./src/LSTM.py 
```

* GRU layer
```
python3 ./src/GRU.py 
```

* Example
  - Run GRU layer with embedded features  
```
python3 ./src/GRU.py -e -i path/to/input  

Options are same for both programs

-i: path to input file.
-p: hour for prediction (default:24) 
-e: incorporate embedded features (default: False)
-d: incorporate encoded features (default: False)  
```

## **Summary**
* Repeating an example provided for gated recurrent model (GRU) with dropout layer described in
  [Deep Learning with Python](https://bit.ly/346tOkH), the error of [0.27](https://bit.ly/2ZCPkOE)
  where training and validation losses meet after 20 epochs, was the lowest
  mean square error. The model showed similar performance on test dataset.

* [Long short-term memory(LSTM)](https://bit.ly/2ZphayP) 
  was also tested by replacing GRU; however, the performance did not change.

* Add-in: 
  I further explored to optimize a model to avoid overfitting.  I mainly focus on the 
  few things, which are listed below:

   1. Data preparation:
      Around half million data was included for the model buiding and testing. These data were
      divided into train, validation, and test group in 8:1:1 ratio.

   2. Parameter selection:
      The number of offset data (prior) taken from current time was
      modified. Similarly, step size used to avoid the continuous
      consideration of data selection was modified. Although 24 hours
      prediction was tested as used in the book, the model showed the best
      performance while testing for 12 hours. Others prediction can also be tested.

   3. Incorporate temporal data:
      Temporal information was excluded in the example shown in the book. However,
      this information was incorporated in two different ways. First, day,
      month, and time were converted into numerical data in such a way that January
      and December are close to each other. Second, using embedded layer from
      Keras, four columns - year, month, day, and time were embedded and
      concatenated with numeric attributes.

   4. Design new model:
      Since GRU with multiple layers showed the best performance in the book,
      I used similar model with different dropout schemes. 

## **Performance**
   - Configuration: 20 epochs with 200 steps/epoch with 0.03 dropouts and recurrent dropouts
   - Features:
      * Default feature indicates 14 different attributes that have numerical
        value.
      * -e indicates embedded features that were combined with 14 attributes.
        They are Day, month, year, and time.
      * -d refers the inclusion of encoded day, month, and time with 14
        attributes. For example, month is encoded in such a way that January and 
        December are very close to each other.

   - 24 hours (default) weather prediction summary
     * GRU and LSTM with three different features were tested.
       They yielded mean square error (MSE) of 0.12 on test dataset.
     * In addition, multiple layers of GRU and LSTM also achieve similar
       performance. These two models with [embedded feature](https://bit.ly/2zuTkSD)
       were selected to check trend of loss function.

   - [Models](https://bit.ly/30LqDgj) for 12 hours prediction
     * Among the previously tested models, LSTM models with embedded and default features with stacking layers
       were selected. Using similar configuration, the model was generated to predict 12 hours' weather. The
       model that included default and embedded features yielded better performance of 0.09 mean squared error.
       This is slightly improved over the same model generated for 24 hours' prediction.
       However, there is no performance improvement (0.12 MSE) for 12 hour's weather prediction over 24 hour's
       when only default feature with similar configuration was used.

       If you downloaded jena_climate_2009_2016.csv in data directory, run the following
       command

       ```
       $ python3 ./src/LSTM.py -e -i ./data/jena_climate_2009_2016.csv -p 12
       ```
     * Similar experiments were also carried out to generated model using GRU.
       Two models yielded 0.09 and 0.10 mean square error with embedded and
       default features respectively.
       ```
       $ python3 ./src/GRU.py -e -i ./data/jena_climate_2009_2016.csv -p 12
       ```


## **Future Direction**
  - The behavior of time series data is highly stocastic, which increases the
    uncertanity in finding a pattern in the data. Nonetheless, recent improvement
    in machine learning algorithm can improves the level of uncertainity and
    provide prediction level with confidence.
  - I personally interested to explore multiple parameters and different models in order to
    optimize the model to yield better performance in the availability of
    computing resources.
  - Moreover, my idea is to expand the utility of a program to employ in the similar time
    series data from different domains.

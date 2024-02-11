# stocky
A Python project to predict S&amp;P 500 returns with TensorFlow and PyTorch.

## Overview

In this project, we aim to take historical data on the S&P 500 index (SPX) in addition to news headlines in order to predict returns minute by minute. We plan to incorporate a sentiment analysis from headlines in order to add another interesting point to the input data which can hopefully play a role in bettering our output. After speaking with Professor Czajka, we determined that a Long Short Term Memory neural network would be the best option in implementing this idea given that we have a sequence of data points over a specific period of time which we will use to train the model. An additional, though not required, goal would be to test the model in real time once it has been trained and tested on the historical data. This of course presents some challenges, such as quickly pulling the index data as well as scraping headlines in real time.

## Data Sets

For this project we will need a large amount of minute by minute stock data on the S&P 500 index along with corresponding newspaper headline data that has been evaluated for sentiment analysis. For this, there are quite a few good datasets from [huggingface](https://huggingface.co/datasets) and [Kaggle](https://www.kaggle.com/datasets) that we will most likely be using. We will need to ensure that the historical S&P 500 index data is cleaned and preprocessed appropriately. This may involve handling missing values, smoothing out noise, and aligning timestamps between the index data and news headlines. The splits for Training, Validation, and Testing will be created using the chosen data set by first splitting all of the data into a training portion and a testing portion. A small portion of the training data for each epoch will be used as validation and the rest will be used to train the model. When training and validating data in a time series model, you can’t use cross-validation with several folds because we want the predictions to be in the future not the past. The model will not be predicting the past in practice so using that in the validation set would not be a good indicator of the model’s success. There are a few methods to validate a time-series model that we can look into and talk with Professor Czajka about to decide which will be best and give the best indicators of the models success. Additionally, if the model performs well, we can begin testing it on the current stock market using the live S&P Index as testing data and checking the accuracy of the model based on the actual curve of the index.

## Network Architecture

As mentioned above, we spoke with Professor Czajka on how to best structure this model and he brought forth the idea of a recurrent neural network since we will have previous time slices impacting the current data being passed to the network. Specifically we are planning to implement a LSTM network which should handle the past data points well for a problem of this nature. Our input data should consist of basic index data such as the open, close, high, low, PE ratio and other measures which help determine the price of the index. Additionally we plan to run sentiment analysis on the headlines to generate a single data point for each time slice which will standardize our input size. The output should consist of a linear activation layer followed by a single neuron that appropriately combines the previous weights into a single data point, the percent return in the next time slice. Our initial thought is to have a time slice of a minute, but we can always make this longer if necessary as we will certainly have enough data to do so.

## Final Thoughts

We acknowledge that predicting returns on pretty much any financial instrument is a difficult endeavor and that our success might be somewhat limited for a project like this. After all, trillions of dollars are poured into hedge funds and quantitative finance firms to arbitrage the market like this. That being said, we hope that implementing sentiment analysis will aid in raising our success. Further, as discussed with Professor Czajka, we may need to swap one of the practical assessments in order to get some practice with recurrent neural networks in time to implement our project.

### Contributions

Walker: Overview, Network Architecture, Final Thoughts

Will: Data Sets, Final Thoughts

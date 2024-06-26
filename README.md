# stocky
A Python project to predict S&amp;P 500 returns with TensorFlow and PyTorch.

# Part 1

## Overview

In this project, we aim to take historical data on the S&P 500 index (SPX) in addition to news headlines to predict returns minute by minute. We plan to incorporate a sentiment analysis from headlines to add another interesting point to the input data which can hopefully play a role in bettering our output. After speaking with Professor Czajka, we determined that a Long Short Term Memory neural network would be the best option in implementing this idea given that we have a sequence of data points over a specific period of time which we will use to train the model. An additional, though not required, goal would be to test the model in real-time once it has been trained and tested on the historical data. This of course presents some challenges, such as quickly pulling the index data as well as scraping headlines in real time.

## Data Sets

For this project, we will need a large amount of stock data on the S&P 500 index along with corresponding newspaper headline data that has been evaluated for sentiment analysis. For this, there are quite a few good datasets from [huggingface](https://huggingface.co/datasets) and [Kaggle](https://www.kaggle.com/datasets) that we will most likely be using. We will need to ensure that the historical S&P 500 index data is cleaned and preprocessed appropriately. This may involve handling missing values, smoothing out noise, and aligning timestamps between the index data and news headlines. The splits for Training, Validation, and Testing will be created using the chosen data set by first splitting all of the data into a training portion and a testing portion. A small portion of the training data for each epoch will be used as validation and the rest will be used to train the model. When training and validating data in a time series model, you can’t use cross-validation with several folds because we want the predictions to be in the future not the past. The model will not be predicting the past in practice so using that in the validation set would not be a good indicator of the model’s success. There are a few methods to validate a time-series model that we can look into and talk with Professor Czajka about to decide which will be best and give the best indicators of the model's success. Additionally, if the model performs well, we can begin testing it on the current stock market using the live S&P Index as testing data and checking the accuracy of the model based on the actual curve of the index.

## Network Architecture

As mentioned above, we spoke with Professor Czajka on how to best structure this model and he brought forth the idea of a recurrent neural network since we will have previous time slices impacting the current data being passed to the network. Specifically, we are planning to implement an LSTM network which should handle the past data points well for a problem of this nature. Our input data should consist of basic index data such as the open, close, high, low, PE ratio, and other measures that help determine the price of the index. Additionally, we plan to run sentiment analysis on the headlines to generate a single data point for each time slice which will standardize our input size. The output should consist of a linear activation layer followed by a single neuron that appropriately combines the previous weights into a single data point, the percent return in the next time slice. Our initial thought is to have a time slice of a minute, but we can always make this longer if necessary as we will certainly have enough data to do so.

## Final Thoughts

We acknowledge that predicting returns on pretty much any financial instrument is a difficult endeavor and that our success might be somewhat limited for a project like this. After all, trillions of dollars are poured into hedge funds and quantitative finance firms to arbitrage the market like this. That being said, we hope that implementing sentiment analysis will aid in raising our success. Further, as discussed with Professor Czajka, we may need to swap one of the practical assessments in order to get some practice with recurrent neural networks in time to implement our project.

### Contributions

Walker: Overview, Network Architecture, Final Thoughts

Will: Data Sets, Final Thoughts

# Part 2

## Sources

For this project, we decided that based on the availability of minute-by-minute stock data, it would be more feasible to look rather at daily stock price data. As a result, we will be using daily open/close, high/low, and % change data downloaded from [investing.com](https://www.investing.com/indices/us-spx-500-historical-data). As far as the news articles for sentiment analysis to input into the model, we will be using a [dataset from Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews) that has 25 news headlines for each day from 2008-06-08 to 2016-07-01 from the subreddit r/worldnews. The data from investing.com also uses this same date range.

## Training splits

For the training data, we will be using the first 80 percent of the data for training with the last 20 percent being the testing set. Eventually, we can try to predict future values with current index prices but for now, we will be using the datasets for testing. Because this is a time-series model, it wouldn't make much sense to do a traditional cross-fold validation as we want to train on data contiguously. Because of this, we will be doing a 5-fold time-series split for cross-validation where the validation set is always in the future. The main difference between the training and validation sections in this case will just be the fact that the validation set will always be the next future values directly after the training portion. The image below illustrates how it will be done.
![image](https://github.com/walkerbagley/stocky/assets/123012662/a8314fa9-a87f-49e7-a3cf-3128498fbc0d)

## Observations.

The dataset for the S&P index values has one value per day (excluding weekends) in the range of dates so as a result has 2032 unique rows of data. The dataset with Reddit headlines has the top 25 news headlines on r/worldnews for every day in the range of dates so it has 73608 unique rows.

## Sample Dataset Entries

- Index Data:
- Date:  6/9/2008  Price: 1,361.76  Open: 1,360.83  High: 1,370.63  Low: 1,350.62  Change %: 0.08%

- News Data:
- Date: 6/8/2008  News: b'Marriage, they said, was reduced to the status of a commercial transaction in which women could be discarded by husbands claiming to have discovered hidden defects in them.'

# Part 3

## How to Run

To run a trained model on the testing data, open [model_testing.ipynb](model_testing.ipynb) in Google Colabs. Then download the [trained model checkpoint](Models/Price-50-100E-0_9814.keras), the [testing dataset](Datasets/Processed Data/normalized_testing_data.csv), and our [customLSTM Module](lstm.py). Upload these to the Colab runtime and run all lines.

## Network Architecture

The Architecture being used for this first model is a Neural Network set up with Keras which is a high level API for TensorFlow. A LSTM Network looked to be our best option for our task of regression on the curve of the S&P 500 index due to the fact that it is a complex time series and storing a running memory on the past points is very important. Stock data can be very influenced by past patterns from a long time ago and while an RNN could be sufficient for shorter sequences, it would not be able to store as much long range data and could suffer more from a vanishing or exploding gradient. This network includes one LSTM layer with 50 units using a relu activation function connected to a one unit dense layer that has no activation function. We use 50 units for the LSTM layer due to the fact that we are looking back 50 time steps (days in this case) in the data and having one unit per time step seemed to make the most sense. The relu activation function is used for the LSTM layer due to the fact that it is efficient when calculating gradients and gave us the best accuracy scores so far. We are using the adam optimizer as it gives us much better training and testing accuracies. 

## Classification Accuracy

As this project seeks to quantify data as accurately as possible rather than categorize data into discrete groups, judging accuracy is far more complex than just finding the number of tests classified correctly. We want to judge how close our model is to actual historical data as opposed to a binary correct or incorrect classification. For this reason, accuracy should be based on minimizing error rather than maximizing correctness. We use mean squared error as our loss function, so it seems redundant to use that for accuracy. A reasonable method we found to measure accuracy was the r2 score. The r2 score is used as a metric for what amount of the relationship between features and output can be explained by the model, which is more along the lines of what we are looking for.

When we trained our model for 100 epochs, we got an r2 score of 0.9992, which is pretty incredible given that r2 maxes out at 1. Obviously the model was very fit to the training data, which given the stochasticity of the stock market, means that we would expect a decently worse r2 score using the testing data. We found this to be a nonfactor, achieving an r2 score of 0.9858, which is pretty incredible. We also saw diminishing returns in terms of performance as we trained with more epochs, possibly due to overfitting.

## Potential Improvements

There are still numerous pieces of this model to work on in the coming weeks. In stock prediction models, it is common to see the model perform well because it can easily mimic the previous day’s output, resulting in a “lag”. We are curious to see if this is an issue with our model and would like to investigate by having the model use previous predictions as input for historical data rather than the historical data itself. This would eliminate any chance for the model to follow along with the actual data and possibly more wholly show whether or not the model is good. In addition to this, we would like to experiment with and optimize the number and types of layers we use in the network to better understand what works well and what doesn’t. 

We are also going to be looking more deeply at which inputs to the network lead to best performance as some of the sentiment data may be unnecessary and we may be able to whittle it down to the sentiment of the most popular article or maybe just the average itself to provide the network with more relevant information. We can additionally mix and match the optimizers and loss functions more thoroughly and observe the test accuracies as a result. We can also see if predicting the change percentage in the closing price yields higher accuracies than just plotting the next point on the price curve. Lastly, we would like to be able to denormalize the data on the curve so that we can output real prices and use it more realistically as a result.

### Contributions

Walker: LSTM library, Performance Metrics, Report

Will: Initial Model Architecture, Model Testing, Results Plotting, Report

# Part 4

## How to run

To run a trained model on the testing data, open [model_testing.ipynb](model_testing.ipynb) in Google Colab. Then download the [trained model checkpoint](Models/Price-5-200E-0_8195.keras), the [testing dataset](<Datasets/Processed Data/normalized_testing_data.csv>), and our [customLSTM Module](lstm.py). Upload these to the Colab runtime and run all lines.

## Test Database

For the test database, it was hard to find a database with a large amount of news headline data over a long period of time. However, we didn’t want to use the same dataset and just make a test split from that as it would be hard to tell how the model would generalize to completely new data. As a result, we decided to compile some news headlines and stock data together to create a new testing dataset to use. We did this by taking all English headlines from this [kaggle dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset) and taking only 5 of the headlines from each day to perform Vader sentiment analysis on. Then, based on the dates from the dataset, we collected price information on the S&P 500 index fund from publicly available data on [investing.com](https://www.investing.com/indices/us-spx-500-historical-data) and then added all of the sentiment scores to the corresponding dates. 

The process for creating this dataset was very similar to how we created the training and validation sets, however, the amount of dates available from this dataset was not very many and as a result, the testing dataset is around a month's worth of stock and corresponding news article data. This testing set is quite small and in making the dataset we realized that to use a model like this, someone may not have a lot of past sentiment or stock data. Because of this and because of the fact that the model still performed quite well on a smaller number of backward time steps, we decided to train the model only using 5 time steps which would represent one business week. As a result, we hoped this would help the model better generalize to newer data where there is not a lot of history to work with at times. 

The main difference between this testing set and the training and validation set is the fact that we used news headlines not just from Reddit and the dates we chose were not continuous from where the training and validation left off. I believe these differences are significant enough to see how well the model generalizes because the stock data can always be grabbed and a user can always manually store news headline data even if they are not doing so from Reddit.

## Results

### Accuracy

After making some changes to the previous model by reducing the number of timesteps to n=5 and changing the number of hidden units to be equal to 2n per the advice of Professor Czajka, we trained the model for 200 epochs and saw a training R2 score of 0.999 and a Validation R2 score of 0.994. 

**Validation Plot**

![validation_plot](https://github.com/walkerbagley/stocky/assets/123012662/3344e044-992b-4b4b-b83c-30f0265c999f)

The testing set saw quite a bit of drop-off in its accuracy as the model achieved an R2 score of 0.8195. 

**Testing Plot**

![testing_plot](https://github.com/walkerbagley/stocky/assets/123012662/5bd38eb7-6bea-4ff4-9195-ccfc029bb4af)

### Interpreting Results and Potential Improvements


While the accuracy was certainly nowhere near as good as on the training and validation sets, it still exceeded our expectations. 

This may be due to a few reasons, firstly:

**Lack of Training-Testing Continuity**
 * With a time series such as stock data, being able to have your training set carry over directly into the test set that you are trying to predict can be very helpful as there are many patterns that can be learned from the stock that will be lost if there is a gap between the two. Ideally, we would want to train this model up until the current day and then begin trying to predict the price of the next day. However, this is obviously harder than it seems and requires a ton of data to be accumulated before trying to make it work. On the opposite end of the spectrum, it could be more helpful to have a model that may not be as accurate but can generalize well to any timeline even without continuity from what it was trained on. Because of these things, it is not immediately clear what would be better but I think it seems pretty clear that the testing accuracy would certainly be higher if the dates started right after the training set dates ended.

**Here is an example of what we would want to do**

![splits](https://github.com/walkerbagley/stocky/assets/123012662/67c7f649-34fb-430b-af95-8b1c585b00ba)

                    
**Smaller Testing Dataset**
* Because the dataset is small and we knew that it might be going into this, we decided to have fewer time steps before each prediction. This fact may mean that because the model isn’t able to look back as far, it is not getting as much helpful information about potential changes. The small number of time steps seemed to work well for the training set but there could be a potential for this to not translate well if the data is disjoint from the testing data in terms of time. In that case, it might actually generalize better with a higher number of steps to look back on.

**Different News Data**
* When looking at the two different data sets for news sentiment, it was pretty obvious to me that the Reddit dataset had a much better quality of headlines than the World News one did. This is because the world news dataset had a lot of different countries and languages of news and as a result, we had to cut down a lot of data points in order to get the readable English entries. Also, we just chose the first 5 news articles for each day whereas the Reddit data set used the top 5 articles per day in terms of the number of upvotes. This means the Reddit dataset could have had more meaningful articles. Along with the meaningfulness of the articles, the Reddit headlines were much more descriptive than the descriptions in the world news dataset which could make a large difference. In the future, we would want to get all our data from the same sources in the same way but obviously, this is not always feasible. 

Overall, our main improvements to this model would be to be able to train on a larger dataset that leads straight into the data we are trying to predict using all the data from the same source.

### Contributions

Walker: Hyperparameter Tuning, Model Training, Report

Will: Testing Set Curation, Testing Set Pre-processing, Report


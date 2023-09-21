# Predicting House Values using KNN, Regression Tree &amp; Regression Line in R
**Goal of the Project:** Predict House Values based on its features employing KNN, Regression Tree & Regression Line. Use appropriate evaluation metrics to measure the performance of the aforementioned methods. 
## Summary
I. Concepts

II. Build Models to Predict House Values
1. Preparation
2. Data Exploration & Preprocessing
3. Predict House Values through KNN 
4. Predict House Values through Regression Tree
5. Predict House Values through Regression Line  
6. Compare three models
## I. Concepts

**KNN for numerical prediction**: "*non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point."*

**Regression Tree**: constructs a tree model to predict the house value based on the decision rules. It recursively splits the data into subsets based on the info variables.

**Regression Line**: fits a linear equation to a set of data point. It predicts house value based on one or more predictor variable.

#### Evaluating Performance

**Mean Absolute Error (MAE)**: gives the magnitute of average absolute error. However, the direction of error is lost.

**Mean Absolute Percentage Error (MAPE)**: gives the percentage score of how much prediction deviate on average from actual values (for example, predictions deviate 18% from actual values from any direction)

**Root Mean Square Error (RMSE)**: penalizes large error. Useful if large errors are undesirable. It has the same unit of the RMSE (if unit of X is $ then RMSE unit is $)

**R-Squared**: measure how close the data are to the fitted regression line.

## II. Build Models to Predict House Values
### 1. Preparation
Prepare to load these libraries:

```
library(caret)
library(rpart.plot)
library(gridExtra)
library(dplyr)
```
### 2. Data Exploration and Preprocessing
First, check through the data set in Excel. Notice that there are a couple of blank rows under total_bedrooms. Since these rows are very small compared to the whole data set (200/20,000), it is okay to get rid of these rows. In other ways, you can also fill these blank rows with the average. But if you choose to get rid of these rows, here is how:

```
df <- read.csv("housevalue.csv")
df <- df[complete.cases(df), ]
```
You can also create a histogram to see the median house value distribution. According to this histogram, the data is skewed to the right with a spike around the $500,000. 

<img width="525" alt="dis" src="https://github.com/codingispink/Predict-House-Values-With-KNN-Regression-Tree-and-Regression-Line/assets/138828365/0a8effa6-b372-4a10-9790-0365c36676e0">


Next, you can examine the relationship between features and the median house value. Instead of comparing each feature to the house value, we can do as below to compare all of them at once:
``` {r }
pairs(df[,c(1,2,3,4,6)], col = "darkgreen")
```

Compare the ocean_proximity with median house value:

```{r}
ggplot(data=df, aes(x=as.factor(ocean_proximity), y=median_house_value)) +
  geom_boxplot() +xlab("ocean_proximity")
```

**Normalization**
Time to normalize the data. Let's grab all values except for the median housing value. Then, with createDataPartition, we split the data into training and testing (70% training and 30% testing). Do this for both the original data and standardized data.

```
df_stand <-df
df_stand[,1:9] <- lapply(df[,1:9], function(x) if(is.numeric(x)){
  scale(x, center=TRUE, scale=TRUE)
} else x)
trainRows <- createDataPartition(y = df_stand$median_house_value, p = 0.7, list = FALSE)

train_set_stand <- df_stand[trainRows,]
test_set_stand <- df_stand[-trainRows,]

train_set <-df[trainRows,]
test_set <- df[-trainRows,]
```
### 3. Predict House Values through KNN 
We can train the model on the standardized data set using the function train(). View the result and make sure it is treated as numerical prediction. Then we can predict the house values on the test_set_stand.
```
knn_model <- train(median_house_value~., train_set_stand, method = "knn")
knn_model
knnPred <- predict(knn_model, test_set_stand)
```

Now save the prediction to the original data set.
```
test_set$knnmedian_house_value <- knnPred
```

**Visualize the prediction results**

*House Value vs KNN Predicted House Value*


<img width="525" alt="hvl" src="https://github.com/codingispink/Predict-House-Values-With-KNN-Regression-Tree-and-Regression-Line/assets/138828365/64b49e50-c6d1-429d-8df1-43057cd04806">


Overall, we see a positive correlation between the house value and KNN predicted house value, which is a really good thing. You can also visualize the error (predict - real) vs real house value.

**Calculate the Predictive Performance Metrics via PostResample**
You can use postResample(pred, obs) to calculate RSME, RSquared, and MAE. MAPE will be the only formula you have to manually calculate.

**a. Obtain the performance metrics via postResample**
```
metrics <- postResample(pred=test_set$knnmedian_house_value, obs=test_set$median_house_value)
metrics
```

**b. Calculate MAPE**
```{r }
MAPE <-mean(abs((test_set$median_house_value-test_set$knnmedian_house_value)/test_set$median_house_value))
names(MAPE) < "MAPE"
MAPE
```
**c. Combine those two metrics**
```{r }
knnMetrics <- c(metrics, MAPE)
options(scipen =999) #avoid scientific notation
knnMetrics
```
### 4. Predict House Values through Regression Tree
Unlike the other method, it is unnecessary to normalize data for regression tree and regression line. Therefore, we will train the regression tree with the non-standardized data set.
**Train the regression tree using the non standardized data**
```{r }
rtree <- train(median_house_value~., train_set, method ='rpart')
rtree
```
**Plot the tree**
``` {r }
rpart.plot(rtree$finalModel, digit =-3)
```

<img width="525" alt="treeplot" src="https://github.com/codingispink/Predict-House-Values-With-KNN-Regression-Tree-and-Regression-Line/assets/138828365/35d25740-81ef-4f2d-a107-2865365bfb02">


**Interpretation** 
In this regression tree, the model predicts that those with median income less than $50,500 (5.05 in original data) will  be likely to purchase a house that is around $173,625 while those with higher salary will be likety to purchase a house that costs up to $332,732. In the group of buyers that get paid less than $50,500, those that purchases house inland will likely buy house up to $112,324 while the others will buy house up to $208,280.

**Save the prediction results**
```{r }
treePred <- predict(rtree, test_set)
test_set$treemedian_house_value <- treePred
```

**Visualize**
```{r }
ggplot(data=test_set, aes(x= median_house_value, y= treemedian_house_value)) +
  geom_point(colour="red") +
  xlim(0,500000) + ylim(0,500000) +
  labs(x="House Value", y ="Decision Tree Predicted House Value")
```

<img width="525" alt="treehvl" src="https://github.com/codingispink/Predict-House-Values-With-KNN-Regression-Tree-and-Regression-Line/assets/138828365/5ada6f0c-a44b-4d68-a516-99b29b33aaab">


**Compute performance metrics like above**

### 5. Predict House Values through Regression Line  
**Train the Linear Regression using "lm"**
``` {r }
lin_reg <- train(median_house_value~., train_set, method= "lm")
lin_reg
```
**Summarize the final model**
Doing so will help you observe which factors contribute positive/negatively to the house value and which one are significant contributors of the final house value.
```{r }
options(scipen=999)
summary(lin_reg$finalModel)
```
**Predict the house values using the testing data**
```{r }
test_set$lmmedian_house_value <- predict(lin_reg, newdata = test_set)
```
**Visualize the prediction results**

<img width="525" alt="linear" src="https://github.com/codingispink/Predict-House-Values-With-KNN-Regression-Tree-and-Regression-Line/assets/138828365/283acdbf-8d16-4434-b7d6-73835e53e15f">

Similarly, we can see that there is a positive correlation between real house value vs Linear Regression predicted house value, which means that the house value tends to be close to the linear regression predicted house value. Just to make sure that you can measure the accuracy, calcualate the metrics with postResample().

### 6. COMPARE THE MODEL 
Use rbind(df,row) to combine existing named vectors into a matrix. 

```{r }
metrics <- rbind(NULL, knnMetrics)
metrics <- rbind(metrics, treeMetrics)
metrics <- rbind(metrics, lmMetrics)
metrics 
```

**The result**
```
                 RMSE  Rsquared      MAE          
knnMetrics  59358.97 0.7329569 40174.04 0.2181261
treeMetrics 86297.93 0.4328938 65225.74 0.3835381
lmMetrics   67463.80 0.6533610 49119.17 0.3835381
```

#### Conclusion
According to the result above, the best model to predict house values is the KNN. KNN has the lowest RMSE, MAE and MAPE. On the other hand, Regression Tree is the worst model.

## ACKNOWLEDGEMENT

[1] California Housing Price Prediction: https://www.kaggle.com/code/suprabhatsk/california-housing-prices-prediction/input?select=housing.csv

[2] Professor De Liu, Lab 5-1 Numerical Prediction, IDSC 4444 - Descriptive and Predictive Analytics

[3] California Housing Value Prediction: https://www.kaggle.com/code/anuvrat29/california-housing-value-prediction

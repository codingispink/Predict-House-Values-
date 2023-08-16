# Predicting House Values using KNN, Regression Tree &amp; Regression Line in R
## Summary
I. Concepts

II. Build Models to Predict House Values
1. Preparation
2. Data Exploration & Preprocessing
3. Predict House Values through KNN Clustering
4. Predict House Values through Regression Tree
5. Predict House Values through Regression Line  
6. Compare three models
## I. Concepts

**KNN**: predicts the house value based on the label of its neighbor. KNN assumes that similar data points lie closer in spatial coordinates.

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


<img width="300" alt="000012" src="https://github.com/codingispink/Predict-House-Values-/assets/138828365/1f9c860a-54f2-48e6-9985-05fc909b5d54">


Next, you can examine the relationship between features and the median house value. Instead of comparing each feature to the house value, we can do as below to compare all of them at once:
``` {r }
pairs(df[,c(1,2,3,4,6)], col = "darkgreen")
```

Since the ocean_proximity (or proximity) is a categorical column with texts, we can't really compare them like above. Therefore, I am going to turn these texts into numbers and compare them separately:

```
#This has already been updated in Excel
<1H OCEAN = 1
INLAND = 2
ISLAND = 3
NEAR BAY =4
NEAR OCEAN =5
```
Then, compare the ocean_proximity with median house value:

```{r}
ggplot(data=df, aes(x=as.factor(proximity), y=median_house_value)) +
  geom_boxplot() +xlab("ocean_proximity")
```

**Normalization**
Time to normalize the data. Let's grab all values except for the median housing value. Then, with createDataPartition, we split the data into training and testing (70% training and 30% testing). Do this for both the original data and standardized data.

```
df_stand <-df
df_stand[,1:7] <- apply(df[,1:7], 2, scale)
trainRows <- createDataPartition(y = df_stand$median_house_value, p = 0.7, list = FALSE)

train_set_stand <- df_stand[trainRows,]
test_set_stand <- df_stand[-trainRows,]

train_set <-df[trainRows,]
test_set <- df[-trainRows,]
```
### 3. Predict House Values through KNN Clustering
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
![000022](https://github.com/codingispink/Predict-House-Values-/assets/138828365/9b70e993-3725-448b-b42e-f7b369c536da)

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

**Train the regression tree using the non standardized data**
```{r }
rtree <- train(median_house_value~., train_set, method ='rpart')
rtree
```
**Plot the tree**
``` {r }
rpart.plot(rtree$finalModel, digit =-3)
```
![000012](https://github.com/codingispink/Predict-House-Values-/assets/138828365/cee4a5ce-2dd6-46d3-a551-3879fad23223)

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
![000010](https://github.com/codingispink/Predict-House-Values-/assets/138828365/e7e0a6b8-60d4-4f92-ab7c-94af07fd0aff)

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
![000010](https://github.com/codingispink/Predict-House-Values-/assets/138828365/262cfac1-3a18-4073-a177-b0ae838d6ece)

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
                RMSE  Rsquared      MAE      MAPE    
knnMetrics  64713.06 0.6837161 44839.00 0.2517936
treeMetrics 90714.38 0.3767587 69885.75 0.4316204
lmMetrics   75397.80 0.5696283 55730.46 0.4316204
```

#### Conclusion
According to the result above, the best model is the KNN. KNN has the lowest RMSE, MAE and MAPE. On the other hand, Regression Tree is the worst model.

## ACKNOWLEDGEMENT

[1] California Housing Price Prediction: https://www.kaggle.com/code/suprabhatsk/california-housing-prices-prediction/input?select=housing.csv

[2] Professor De Liu, Lab 5-1 Numerical Prediction, IDSC 4444 - Descriptive and Predictive Analytics

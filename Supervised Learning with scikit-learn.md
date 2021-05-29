# Supervised Learning with scikit-learn

## Classification

# Supervised learning

## 1. Supervised learning

Welcome to the course! My name is Andy and I'm a core contributor and co-maintainer of scikit-learn. You're here to learn about the wonderful world of supervised learning, arguably the most important branch of machine learning.

## 2. What is machine learning?

But what is machine learning? **Machine learning is the science and art of giving computers the ability to learn to make decisions from data without being explicitly programmed.** For example, your computer can learn to predict whether an email is spam or not spam given its content and sender. Another example: your computer can learn to cluster, say, Wikipedia entries, into different categories based on the words they contain. It could then assign any new Wikipedia article to one of the existing clusters. Notice that, in the first example, we are trying to predict a particular class label, that is, spam or not spam. In the second example, there is no such label. **When there are labels present, we call it supervised learning. When there are no labels present, we call it unsupervised learning.**

## 3. Unsupervised learning

**Unsupervised learning, in essence, is the machine learning task of uncovering hidden patterns and structures from unlabeled data.** For example, a business may wish to group its customers into distinct categories based on their purchasing behavior without knowing in advance what these categories maybe. This is known as clustering, one branch of unsupervised learning.

## 4. Reinforcement learning

There is also reinforcement learning, in which machines or software agents interact with an environment. Reinforcement agents are able to **automatically figure out how to optimize their behavior given a system of rewards and punishments.** Reinforcement learning **draws inspiration from behavioral psychology** and has applications in many fields, such as, economics, genetics, as well as game playing. In 2016, reinforcement learning was used to train Google DeepMind's AlphaGo, which was the first computer program to beat the world champion in Go.

## 5. Supervised learning

But let's come back to supervised learning, which will be the focus of this course. In supervised learning, we have several data points or samples, described using predictor variables or features and a target variable. **Our data is commonly represented in a table structure** such as the one you see here, in which there is a row for each data point and a column for each feature. Here, we see the iris dataset: each row represents measurements of a different flower and each column is a particular kind of measurement, like the width and length of a certain part of the flower. The aim of supervised learning is to build a model that is able to predict the target variable, here the particular species of a flower, given the predictor variables, here the physical measurements. **If the target variable consists of categories,** like 'click' or 'no click', 'spam' or 'not spam', or different species of flowers, **we call the learning task classification**. Alternatively, **if the target is a continuously varying variable**, for example, the price of a house, **it is a regression task**. In this chapter, we will focus on classification. In the following, on regression.

## 6. Naming conventions

A note on naming conventions: out in the wild, you will find that what we call a **feature**, others may call a **predictor variable** or **independent variable,** and what we call the **target variable**, others may call **dependent variable** or **response variable.**

## 7. Supervised learning

The goal of supervised learning is frequently to either automate a time-consuming or expensive manual task, such as a doctor's diagnosis, or to make predictions about the future, say w hether a customer will click on an add or not. For supervised learning, you need labeled data and there are many ways to get it: you can get historical data, which already has labels that you are interested in; you can perform experiments to get labeled data, such as A/B-testing to see how many clicks you get; or you can also crowdsourced labeling data which, like reCAPTCHA does for text recognition. In any case, the goal is to learn from data for which the right output is known, so that we can make predictions on new data for which we don't know the output.

## 8. Supervised learning in Python

There are many ways to perform supervised learning in Python. In this course, we will use scikit-learn, or sklearn, one of the most popular and user-friendly machine learning libraries for Python. It also integrates very well with the SciPy stack, including libraries such as NumPy. There are a number of other ML libraries out there, such as TensorFlow and keras, which are well worth checking out once you got the basics down.

## 9. Let's practice!

Let's now jump into an exercise!



# Exploratory data analysis

**Got It!**

## 1. Exploratory data analysis

Let's now jump into our first dataset.

## 2. The Iris dataset

It contains data pertaining to iris flowers in which the features consist of four measurements: petal length, petal width, sepal length, and sepal width. The target variable encodes the species of flower and there are three possibilities: 'versicolor', 'virginica', and 'setosa'. As this is one of the datasets included in scikit-learn,

## 3. The Iris dataset in scikit-learn

we'll import it from there with from sklearn import datasets. In the exercises, you'll get practice at importing files from your local file system for supervised learning. We'll also import pandas, numpy, and pyplot under their standard aliases. In addition, we'll set the plotting style to ggplot using plt dot style dot use. Firstly, because it looks great and secondly, in order to help all you R aficionados feel at home. We then load the dataset with datasets dot load iris and assign the data to a variable iris. Checking out the type of iris, we see that it's a bunch, which is similar to a dictionary in that it contains key-value pairs. Printing the keys, we see that they are the feature names: DESCR, which provides a description of the dataset; the target names; the data, which contains the values features; and the target, which is the target data.

## 4. The Iris dataset in scikit-learn

As you see here, both the feature and target data are provided as NumPy arrays. The dot shape attribute of the array feature array tells us that there are 150 rows and four columns. Remember: samples are in rows, features are in columns. Thus we have 150 samples and the four features: petal length and width and sepal length and width. Moreover, note that the target variable is encoded as zero for "setosa", 1 for "versicolor" and 2 for "virginica". We can see this by printing iris dot target names, in which "setosa" corresponds to index 0, "versicolor" to index 1 and "virginica" to index 2.

## 5. Exploratory data analysis (EDA)

In order to perform some initial exploratory data analysis, or EDA for short, we'll assign the feature and target data to X and y, respectively. We'll then build a DataFrame of the feature data using pd dot DataFrame and also passing column names. Viewing the head of the data frame shows us the first five rows.

## 6. Visual EDA

Now, we'll do a bit of visual EDA. We use the pandas function scatter matrix to visualize our dataset. We pass it the our DataFrame, along with our target variable as argument to the parameter c, which stands for color, ensuring that our data points in our figure will be colored by their species. We also pass a list to fig size, which specifies the size of our figure, as well as a marker size and shape.

## 7. Visual EDA

The result is a matrix of figures, which on the diagonal are histograms of the features corresponding to the row and column. The off-diagonal figures are scatter plots of the column feature versus row feature colored by the target variable. There is a great deal of information in this scatter matrix.

## 8. Visual EDA

See, here for example, that petal width and length are highly correlated, as you may expect, and that flowers are clustered according to species.

## 9. Let's practice!

Now it's your turn to dive into a few exercises and to do some EDA. Then we'll back to do some machine learning. Enjoy!





# The classification challenge

**Got It!**

## 1. The classification challenge

We have a set of labeled data and we want to build a classifier that takes unlabeled data as input and outputs a label. So how do we construct this classifier? We first need choose a type of classifier and it needs to learn from the already labeled data. For this reason, we call the already labeled data the training data. So let's build our first classifier!

## 2. k-Nearest Neighbors

We'll choose a simple algorithm called K-nearest neighbors. The basic idea of K-nearest neighbors, or KNN, is to predict the label of any data point by looking at the K, for example, 3, closest labeled data points and getting them to vote on what label the unlabeled point should have.

## 3. k-Nearest Neighbors

In this image, there's an example of KNN in two dimensions: how do you classify the data point in the middle?

## 4. k-Nearest Neighbors

Well, if k equals 3,

## 5. k-Nearest Neighbors

you would classify it as red and, if k equals 5, as green.

## 6. k-Nearest Neighbors

and, if k equals 5,

## 7. k-Nearest Neighbors

as green.

## 8. k-NN: Intuition

To get a bit of intuition for KNN, let's check out a scatter plot of two dimensions of the iris dataset, petal length and petal width. The following holds for higher dimensions, however, we'll show the 2D case for illustrative purposes.

## 9. k-NN: Intuition

What the KNN algorithm essentially does is create a set of decision boundaries and we visualize the 2D case here.

## 10. k-NN: Intuition

Any new data point here will be predicted 'setosa',

## 11. k-NN: Intuition

any new data point here will be predicted 'virginica',

## 12. k-NN: Intuition

and any new data point here will be predicted 'versicolor'.

## 13. Scikit-learn fit and predict

All machine learning models in scikit-learn are implemented as python classes. These classes serve two purposes: they implement the algorithms for learning a model, and predicting, while also storing all the information that is learned from the data. Training a model on the data is also called fitting the model to the data. In scikit-learn, we use the fit method to do this. Similarly, the predict method is what we use to predict the label of an, unlabeled data point.

## 14. Using scikit-learn to fit a classifier

Now we're going to fit our very first classifier using scikit-learn! To do so, we first need to import it. To this end, we import KNeighborsClassifier from sklearn dot neighbors. We then instantiate our KNeighborsClassifier, set the number of neighbors equal to 6, and assign it to the variable knn. Then we can fit this classifier to our training set, the labeled data. To do so, we apply the method fit to the classifier and pass it two arguments: the features as a NumPy array and the labels, or target, as a NumPy array. The scikit-learn API requires firstly that you have the data as a NumPy array or pandas DataFrame. It also requires that the features take on continuous values, such as the price of a house, as opposed to categories, such as 'male' or 'female'. It also requires that there are no missing values in the data. All datasets that we'll work with now satisfy these final two properties. Later in the course, you'll see how to deal with categorical features and missing data. In particular, the scikit-learn API requires that the features are in an array where each column is a feature and each row a different observation or data point. Looking at the shape of iris data, we see that there are 150 observations of four features. Similarly, the target needs to be a single column with the same number of observations as the feature data. We see in this case there are indeed also 150 labels. Also check out what is returned when we fit the classifier: it returns the classifier itself and modifies it to fit it to the data. Now that we have fit our classifier, lets use it to predict on some unlabeled data!

## 15. Predicting on unlabeled data

Here we have set of observations, X new. We use the predict method on the classifier and pass it the data. Once again, the API requires that we pass the data as a NumPy array with features in columns and observations in rows; checking the shape of X new, we see that it has three rows and four columns, that is, three observations and four features. Then we would expect calling knn dot predict of X new to return a three-by-one array with a prediction for each observation or row in X new. And indeed it does! It predicts one, which corresponds to 'versicolor' for the first two observations and 0, which corresponds to 'setosa' for the third.

## 16. Let's practice!









# Measuring model performance

**Got It!**

## 1. Measuring model performance

Now that we know how to fit a classifier and use it to predict the labels of previously unseen data, we need to figure out how to measure its performance. That is, we need a metric.

## 2. Measuring model performance

In classification problems, accuracy is a commonly-used metric. The accuracy of a classifier is defined as the number of correct predictions divided by the total number of data points. This begs the question though: which data do we use to compute accuracy? What we are really interested in is how well our model will perform on new data, that is, samples that the algorithm has never seen before.

## 3. Measuring model performance

Well, you could compute the accuracy on the data you used to fit the classifier. However, as this data was used to train it, the classifier's performance will not be indicative of how well it can generalize to unseen data. For this reason, it is common practice to split your data into two sets, a training set and a test set. You train or fit the classifier on the training set. Then you make predictions on the labeled test set and compare these predictions with the known labels. You then compute the accuracy of your predictions.

## 4. Train/test split

To do this, we first import train test split from sklearn dot model selection. We then use the train test split function to randomly split our data. The first argument will be the feature data, the second the targets or labels. The test size keyword argument specifies what proportion of the original data is used for the test set. Lastly, the random state kwarg sets a seed for the random number generator that splits the data into train and test. Setting the seed with the same argument later will allow you to reproduce the exact split and your downstream results. train test split returns four arrays: the training data, the test data, the training labels, and the test labels. We unpack these into four variables: X train, X test, y train, and y test, respectively. By default, train test split splits the data into 75% training data and 25% test data, which is a good rule of thumb. We specify the size of the test size using the keyword argument test size, which we do here to set it to 30%. It is also best practice to perform your split so that the split reflects the labels on your data. That is, you want the labels to be distributed in train and test sets as they are in the original dataset. To achieve this, we use the keyword argument stratify equals y, where y the list or array containing the labels. We then instantiate our K-nearest neighbors classifier, fit it to the training data using the fit method, make our predictions on the test data and store the results as y pred. Printing them shows that the predictions take on three values, as expected. To check out the accuracy of our model, we use the score method of the model and pass it X test and y test. See here that the accuracy of our K-nearest neighbors model is approximately 95%, which is pretty good for an out-of-the-box model!

## 5. Model complexity

Recall that we recently discussed the concept of a decision boundary. Here, we visualize a decision boundary for several, increasing values of K in a KNN model. Note that, as K increases, the decision boundary gets smoother and less curvy. Therefore, we consider it to be a less complex model than those with a lower K. Generally, complex models run the risk of being sensitive to noise in the specific data that you have, rather than reflecting general trends in the data. This is know as overfitting.

1. 1 Source: Andreas Müller & Sarah Guido, Introduction to Machine Learning with Python

## 6. Model complexity and over/underfitting

If you increase K even more and make the model even simpler, then the model will perform less well on both test and training sets, as indicated in this schematic figure, known as a model complexity curve.

## 7. Model complexity and over/underfitting

This is called underfitting.

## 8. Model complexity and over/underfitting

We can see that there is a sweet spot in the middle that gives us the best performance on the test set.

## 9. Let's practice!

OK, now it's your turn to practice splitting your data, computing accuracy on your test set, and plotting model complexity curves!





# Introduction to regression

## 1. Introduction to regression

Congrats on making it through that introduction to supervised learning and classification. Now, we're going to check out the other type of supervised learning problem: regression. In regression tasks, the target value is a continuously varying variable, such as a country's GDP or the price of a house.

## 2. Boston housing data

Our first regression task will be using the Boston housing dataset! Let's check out the data. First, we load it from a comma-separated values file, also known as a csv file, using pandas' read csv function. See the DataCamp course on importing data for more information on file formats and loading your data. Note that you can also load this data from scikit-learn's built-in datasets. We then view the head of the data frame using the head method. The documentation tells us the feature 'CRIM' is per capita crime rate, 'NX' is nitric oxides concentration, and 'RM' average number of rooms per dwelling, for example. The target variable, 'MEDV', is the median value of owner occupied homes in thousands of dollars.

## 3. Creating feature and target arrays

Now, given data as such, recall that scikit-learn wants 'features' and target' values in distinct arrays, X and y,. Thus, we split our DataFrame: in the first line here, we drop the target; in the second, we keep only the target. Using the values attributes returns the NumPy arrays that we will use.

## 4. Predicting house value from a single feature

As a first task, let's try to predict the price from a single feature: the average number of rooms in a block. To do this, we slice out the number of rooms column of the DataFrame X, which is the fifth column into the variable X rooms. Checking the type of X rooms and y, we see that both are NumPy arrays. To turn them into NumPy arrays of the desired shape, we apply the reshape method to keep the first dimension, but add another dimension of size one to X.

## 5. Plotting house value vs. number of rooms

Now, let's plot house value as a function of number of rooms using matplotlib's plt dot scatter. We'll also label our axes using x label and y label.

## 6. Plotting house value vs. number of rooms

We can immediately see that, as one might expect, more rooms lead to higher prices.

## 7. Fitting a regression model

It's time to fit a regression model to our data. We're going to use a model called linear regression, which we'll explain in the next video. But first, I'm going to show you how to fit it and to plot its predictions. We import numpy as np, linear model from sklearn, and instantiate LinearRegression as regr. We then fit the regression to the data using regr dot fit and passing in the data, the number of rooms, and the target variable, the house price, as we did with the classification problems. After this, we want to check out the regressors predictions over the range of the data. We can achieve that by using np linspace between the maximum and minimum number of rooms and make a prediction for this data.

## 8. Fitting a regression model

Plotting this line with the scatter plot results in the figure you see here.

## 9. Let's practice!





# The basics of linear regression

## 1. The basics of linear regression

Now, how does linear regression actually work?

## 2. Regression mechanics

We want to fit a line to the data and a line in two dimensions is always of the form y = ax + b, where y is the target, x is the single feature, and a and b are the parameters of the model that we want to learn. So the question of fitting is reduced to: how do we choose a and b? A common method is to define an error function for any given line and then to choose the line that minimizes the error function. Such an error function is also called a loss or a cost function.

## 3. The loss function

What will our loss function be? Intuitively, we want the line to be as close to the

## 4. The loss function

actual data points as possible. For this reason, we wish to minimize the vertical distance between the fit and the data. So for each data point,

## 5. The loss function

**we calculate the vertical distance between it and the line. This distance is called a residual.**

## 6. The loss function

Now, we could try to minimize the sum of the residuals,

## 7. The loss function

but then a large positive residual would cancel out

## 8. The loss function

a large negative residual. For this reason we minimize the sum of the squares of the residuals! This will be our loss function and using this loss function is commonly called **ordinary least squares**, or **OLS** for short. Note that this is the same as minimizing the mean squared error of the predictions on the training set. See our statistics curriculum for more detail. When you call fit on a linear regression model in scikit-learn, it performs this OLS under the hood.

## 9. Linear regression in higher dimensions

When we have two features and one target, a line is of the form y = a1x1 + a2x2 + b, so to fit a linear regression model is to specify three variables, a1, a2, and b. In higher dimensions, that is, when we have more than one or two features, a line of this form, so fitting a linear regression model is to specify a coefficient, ai, for each feature, as well as the variable, b. The scikit-learn API works exactly the same in this case: you pass the fit method two arrays: one containing the features, the other the target variable. Let's see this in action.

## 10. Linear regression on all features

In this code, we are working with all the features from the Boston Housing dataset. We split it into training and test sets; we instantiate the regressor, fit it on the training set and predict on the test set. We saw that, in the world of classification, we could use accuracy as a metric of model performance. The default scoring method for linear regression is called R squared. Intuitively, this metric quantifies the amount of variance in the target variable that is predicted from the feature variables. See the scikit-learn documentation and our statistics curriculum for more details. To compute the R squared, we once again apply the method score to the model and pass it two arguments: the test data and the test data target. Note that generally you will never use linear regression out of the box like this; you will most likely wish to use regularization, which we'll see soon and which places further constraints on the model coefficients. However, learning about linear regression and how to use it in scikit-learn is an essential first step toward using regularized linear models.

## 11. Let's practice!









# Cross-validation

## 1. Cross-validation

Great work on those regression challenges! You are now also becoming more acquainted with train test split and computing model performance metrics on your test set. Can you spot a potential pitfall of this process? Well, let's think about it for a bit:

## 2. Cross-validation motivation

if you're computing R squared on your test set, the R squared returned is dependent on the way that you split up the data! The data points in the test set may have some peculiarities that mean the R squared computed on it is not representative of the model's ability to generalize to unseen data. To combat this dependence on what is essentially an arbitrary split, we use a technique called cross-validation.

## 3. Cross-validation basics

We begin by splitting the dataset into five groups or folds.

## 4. Cross-validation basics

Then we **hold out the first fold as a test set**,

## 5. Cross-validation basics

**fit our model on the remaining four folds,**

## 6. Cross-validation basics

predict on the test set, and

## 7. Cross-validation basics

compute the metric of interest.

## 8. Cross-validation basics

Next, we hold out the

## 9. Cross-validation basics

second fold as our test set,

## 10. Cross-validation basics

fit on the remaining data,

## 11. Cross-validation basics

predict on the test set, and

## 12. Cross-validation basics

compute the metric of interest. Then similarly

## 13. Cross-validation basics

with the third,

## 14. Cross-validation basics

fourth, and

## 15. Cross-validation basics

fifth fold.

## 16. Cross-validation basics

As a result we **get five values of R squared** from which we can compute statistics of interest, such as the mean and median and 95% confidence intervals.

## 17. Cross-validation and model performance

As we split the dataset into five folds, we call this process **5-fold cross validation**. If you use 10 folds, it is called 10-fold cross validation. More generally, if you use k folds, it is called k-fold cross validation or k-fold CV. There is, however, a trade-off as using more folds is more computationally expensive. This is because you are fittings and predicting more times. This method avoids the problem of your metric of choice being dependent on the train test split.

## 18. Cross-validation in scikit-learn

To perform k-fold CV in scikit-learn, we first import cross val score from sklearn dot model selection. As always, we instantiate our model, in this case, a regressor. We then call cross val score with the regressor, the feature data, and the target data as the first three positional arguments. We also specify the number of folds with the keyword argument, cv. This returns an array of cross-validation scores, which we assign to cv results. The length of the array is the number of folds utilized. Note that the score reported is R squared, as this is the default score for linear regression. . We print the scores here. We can also, for example, compute the mean, which we also do.

## 19. Let's practice!

Now it's your turn to try your hand at k-fold cross-validation in the interactive exercises. Have fun!





# Regularized regression

**Got It!**

## 1. Regularized regression

Recall that

## 2. Why regularize?

what fitting a linear regression does is **minimize a loss function** to choose a coefficient ai for each feature variable. If we allow these coefficients or parameters to be super large, we can get overfitting. It isn't so easy to see in two dimensions, but when you have loads and loads of features, that is, if your data sit in a high-dimensional space with large coefficients, it gets easy to predict nearly anything. For this reason, it is common practice to alter the loss function so that it **penalizes for large coefficients**. This is called **regularization**. The first type of regularized regression that we'll look at is called **ridge regression**

## 3. Ridge regression

in which our loss function is the standard OLS loss function plus the squared value of each coefficient multiplied by some constant alpha. Thus, when minimizing the loss function to fit to our data, models are penalized for coefficients with a large magnitude: large positive and large negative coefficients, that is. Note that alpha is a parameter we need to choose in order to fit and predict. Essentially, we can select the alpha for which our model performs best. Picking alpha for ridge regression is similar to picking k in KNN. This is called hyperparameter tuning and we'll see much more of this soon. This alpha, which you may also see called lambda in the wild, can be thought of as a parameter that controls model complexity. Notice that when alpha is equal to zero, we get back OLS. Large coefficients in this case are not penalized and the overfitting problem is not accounted for. A very high alpha means that large coefficients are significantly penalized, which can lead to a model that is too simple and ends up underfitting the data. The method of performing ridge regression with scikit-learn mirrors the other models that we have seen.

## 4. Ridge regression in scikit-learn

We import Ridge from sklearn dot linear model, we split our data into test and train, fit on the training, and predict on the test. Note that we set alpha using the keyword argument alpha. Also notice the argument normalize: setting this equal to True ensures that all our variables are on the same scale and we will cover this in more depth later. There is another type of regularized regression called lasso regression,

## 5. Lasso regression

in which our loss function is the standard OLS loss function plus the absolute value of each coefficient multiplied by some constant alpha.

## 6. Lasso regression in scikit-learn

The method of performing lasso regression in scikit-learn mirrors ridge regression, as you can see here.

## 7. Lasso regression for feature selection

One of the really cool aspects of lasso regression is that it can be used to select important features of a dataset. This is because it tends to shrink the coefficients of less important features to be exactly zero. The features whose coefficients are not shrunk to zero are 'selected' by the LASSO algorithm. Let's check this out in practice.

## 8. Lasso for feature selection in scikit-learn

We import Lasso as before and store the feature names in the variable names. We then instantiate our regressor, fit it to the data as always. Then we can extract the coef attribute and store in lasso coef. Plotting the coefficients as a function of feature name yields this figure

## 9. Lasso for feature selection in scikit-learn

and you can see directly that the most important predictor for our target variable, housing price, is number of rooms! This is not surprising and is a great sanity check. This type of feature selection is very important for machine learning in an industry or business setting because it allows you, as a Data Scientist, to communicate important results to non-technical colleagues. And bosses! The power of reporting important features from a linear model cannot be overestimated. It is also valuable in research science, in order identify which factors are important predictors for various physical phenomena.

## 10. Let's practice!





# How good is your model?





**Got It!**

## 1. How good is your model?

In classification,

## 2. Classification metrics

we've seen that you can use accuracy, the fraction of correctly classified samples, to measure model performance. However, accuracy is not always a useful metric.

## 3. Class imbalance example: Emails

Consider a spam classification problem in which 99% of emails are real and only 1% are spam. I could build a model that classifies all emails as real; this model would be correct 99% of the time and thus have an accuracy of 99%, which sounds great. However, this naive classifier does a horrible job of predicting spam: it never predicts spam at all, so it completely fails at its original purpose. The situation when one class is more frequent is called class imbalance because the class of real emails contains way more instances than the class of spam. This is a very common situation in practice and requires a more nuanced metric to assess the performance of our model.

## 4. Diagnosing classification predictions

Given a binary classifier, such as our spam email example, we can draw up a 2-by-2 matrix that summarizes predictive performance called a **confusion matrix**:

## 5. Diagnosing classification predictions

across the top are the predicted labels,

## 6. Diagnosing classification predictions

down the side the actual labels,

## 7. Diagnosing classification predictions

such as you see here. Given any model, we can fill in the confusion matrix according to its predictions.

## 8. Diagnosing classification predictions

In the top left square, we have the number of spam emails correctly labeled;

## 9. Diagnosing classification predictions

in the bottom right square, we have the number of real emails correctly labeled;

## 10. Diagnosing classification predictions

in the top right, the number of spam emails incorrectly labeled;

## 11. Diagnosing classification predictions

and in the bottom left, the number of real emails incorrectly labeled. Note that correctly labeled spam emails

## 12. Diagnosing classification predictions

are referred to as true positives and

## 13. Diagnosing classification predictions

correctly labeled real emails as true negatives.

## 14. Diagnosing classification predictions

While incorrectly labeled spam will be referred to as false negatives and

## 15. Diagnosing classification predictions

incorrectly labeled real emails as false positives.

## 16. Diagnosing classification predictions

Usually, the "class of interest" is called the positive class. As we are trying to detect spam, this makes spam the positive class. Which class you call positive is really up to you. So why do we care about the confusion matrix? First, notice that you can retrieve **accuracy** from the confusion matrix: it's the sum of the diagonal divided by the total sum of the matrix.

## 17. Metrics from the confusion matrix

There are several other important metrics you can easily calculate from the confusion matrix. **Precision**, which is the number of true positives divided by the total number of true positives and false positives. It is also called the positive predictive value or PPV. In our case, this is the number of correctly labeled spam emails divided by the total number of emails classified as spam. **Recall**, which is the number of true positives divided by the total number of true positives and false negatives. This is also called **sensitivity**, hit rate, or true positive rate. The **F1-score** is defined as two times the product of the precision and recall divided by the sum of the precision and recall, in other words, it's the harmonic mean of precision and recall. To put it in plain language, high precision means that our classifier had a low false positive rate, that is, not many real emails were predicted as being spam. Intuitively, high recall means that our classifier predicted most positive or spam emails correctly.

## 18. Confusion matrix in scikit-learn

Let's now compute the confusion matrix, along with the metrics for the classifier we trained on the voting dataset in the exercise. We import classification report and confusion matrix from sklearn dot metrics. As always, we instantiate our classifier, split the data into train and test, fit the training data, and predict the labels of the test set.

## 19. Confusion matrix in scikit-learn

To compute the confusion matrix, we pass the test set labels and the predicted labels to the function confusion matrix. To compute the resulting metrics, we pass the same arguments to classification report, which outputs a string containing all the relevant metrics, which you can see here. For all metrics in scikit-learn, the first argument is always the true label and the prediction is always the second argument.

## 20. Let's practice!







# Logistic regression and the ROC curve

## 1. Logistic regression and the ROC curve

It's time to introduce another model to your classification arsenal: **logistic regression**. Despite its name, **logistic regression is used in classification problems**, not regression problems. We won't go into the mathematical details here, see our stats courses for that, but we will provide an intuition towards how logistic regression or log reg works for binary classification, that is, when we have two possible labels for the target variable.

## 2. Logistic regression for binary classification

Given one feature, **log reg will output a probability**, p, with respect to the target variable. If p is greater than 0 (point) 5, we label the data as '1'; if p is less than 0 (point) 5, we label it '0'.

## 3. Linear decision boundary

Note that log reg produces a linear decision boundary. Using logistic regression in scikit-learn

## 4. Logistic regression in scikit-learn

follows exactly the same formula that you now know so well: perform the necessary imports, instantiate the classifier, split your data into training and test sets, fit the model on your training data, and predict on your test set. Here we have used the voting dataset that you worked with earlier in the course.

## 5. Probability thresholds

Notice that in defining logistic regression, we have specified a threshold of 0 (point) 5 for the probability, a threshold that defines our model. Note that this is not particular for log reg but also could be used for KNN, for example. Now, what happens as we vary this threshold?

## 6. The ROC curve

In particular, what happens to the true positive and false positive rates as we vary the threshold?

## 7. The ROC curve

When the threshold equals zero, the model predicts '1' for all the data, which means the true positive rate is equal to the false positive rate

## 8. The ROC curve

is equal to one. When the threshold

## 9. The ROC curve

equals '1', the model predicts '0' for all data, which means that both true and false positive rates

## 10. The ROC curve

are 0. If we

## 11. The ROC curve

vary the **threshold** between these two extremes, we get a series of different false positive and true positive rates.

## 12. The ROC curve

The set of points we get when trying all possible thresholds is called the receiver operating characteristic curve or ROC curve.

## 13. Plotting the ROC curve

To plot the ROC curve, we import roc curve from sklearn dot metrics; we then call the function roc curve; the first argument is given by the actual labels, the second by the predicted probabilities. A word on this in a second. We unpack the result into three variables: false positive rate, FPR; true positive rate, TPR; and the thresholds. We can then plot the FPR and TPR using pyplot's plot function to produce a figure such as this.

## 14. Plotting the ROC curve

We used the predicted probabilities of the model assigning a value of '1' to the observation in question. This is because to compute the ROC we do not merely want the predictions on the test set, but we want the probability that our log reg model outputs before using a threshold to predict the label. To do this we apply the method predict proba to the model and pass it the test data. predict proba returns an array with two columns: each column contains the probabilities for the respective target values. We choose the second column, the one with index 1, that is, the probabilities of the predicted labels being '1'.

## 15. Let's practice!





# Area under the ROC curve

## 1. Area under the ROC curve

Now the question is: given the ROC curve, can we extract a metric of interest?

## 2. Area under the ROC curve (AUC)

Consider the following: **the larger the area under the ROC curve, the better our model is!** The way to think about this is the following: if we had a model which produced an ROC curve that had a

## 3. Area under the ROC curve (AUC)

single point at 1,0, the upper left corner, representing a true positive rate of one and

## 4. Area under the ROC curve (AUC)

a false positive rate of zero, this would be a great model! For this reason,

## 5. Area under the ROC curve (AUC)

the area under the ROC, commonly denoted as AUC, is another popular metric for classification models.

## 6. AUC in scikit-learn

To compute the AUC, we import roc auc score from sklearn dot metrics. We instantiate our classifier, split our data into train and test sets, and fit the model to the training set. To compute the AUC, we first compute the predicted probabilities as above and then pass the true labels and the predicted probabilities to roc auc score. We can also compute the AUC using cross-validation.

## 7. AUC using cross-validation

To do so, we import and use the function cross val score as before, passing it the estimator, the features, and the target. We then additionally pass it the keyword argument scoring equals "roc auc" and print the AUC list as you can see here.

## 8. Let's practice!

Now, it's your turn to compute AUCs in the interactive exercise.









# Hyperparameter tuning

## 1. Hyperparameter tuning

Welcome back. Now that you're developing a feel for judging how well your models are performing, it is time to supercharge them.

## 2. Hyperparameter tuning

We have seen that when **fitting a linear regression**, what we are really doing is choosing parameters for the model that fit the data the best. We also saw that we had to **choose a value for the alpha in ridge and lasso regression** before fitting it. Analogously, before fitting and predicting **K-nearest neighbors, we need to choose n neighbors**. Such parameters, **ones that need to be specified before fitting a model, are called hyperparameters.** In other words, these are parameters that cannot be explicitly learned by fitting the model. Herein lies a fundamental key for building a successful model:

## 3. Choosing the correct hyperparameter

choosing the correct hyperparameter. The basic idea is to try a whole bunch of different values, fit all of them separately, see how well each performs, and choose the best one! This is called **hyperparameter tuning** and doing so in this fashion is the current standard. There may be a better way, however, and if you figure it out, I'd be surprised if it wouldn't make you famous. Now, **when fitting different values of a hyperparameter, it is essential to use cross-validation as using train test split alone** would risk overfitting the hyperparameter to the test set. We'll see in the next video that, even after **tuning our hyperparameters using cross-validation,** we'll want to have already split off a test set in order to report how well our model can be expected to perform on a dataset that it has never seen before.

## 4. Grid search cross-validation

The basic idea is as follows: we choose a grid of possible values we want to try for the hyperparameter or hyperparameters. For example, if we had two hyperparameters, C and alpha, the grid of values to test could look like this.

## 5. Grid search cross-validation

We then perform k-fold cross-validation for each point in the grid, that is, for each choice of hyperparameter or combination of hyperparameters.

## 6. Grid search cross-validation

We then choose for our model the choice of hyperparameters that performed the best! This is called a grid search and in scikit-learn we implement it using the class GridSearchCV.

## 7. GridSearchCV in scikit-learn

We import GridSearchCV from sklearn dot model selection. We then specify the hyperparameter as a dictionary in which the keys are the hyperparameter names, such as n neighbors in KNN or alpha in lasso regression. See the documentation of each model for the name of its hyperparameters. The values in the grid dictionary are lists containing the values we wish to tune the relevant hyperparameter or hyperparameters over. If we specify multiple parameters, all possible combinations will be tried. As always, we instantiate our classifier. We then use GridSearchCV and pass it our model, the grid we wish to tune over and the number of folds that we wish to use. This returns a GridSearch object that you can then fit to the data and this fit performs the actual grid search inplace. We can then apply the attributes best params and best score, respectively, to retrieve the hyperparameters that perform the best along with the mean cross-validation score over that fold.

## 8. Let's practice!

Now, it's your turn to practice your new grid search cross-validation chops. You'll also learn about RandomizedSearchCV, which is similar to GridSearch, except that it is able to jump around the grid. Happy grid searching!!



# Hold-out set for final evaluation

## 1. Hold-out set for final evaluation

Congrats on making it through those exercises. I hope that you had some serious fun with **GridSearchCV** and **RandomizedSearchCV**. After using K-fold cross-validation to tune my model's hyperparameters,

## 2. Hold-out set reasoning

I may want to report how well my model can be expected to perform on a dataset that it has never seen before, given my scoring function of choice. So, I want to use my model to predict on some labeled data, compare my prediction to the actual labels, and compute the scoring function. However, if I have used all of my data for cross-validation, estimating my model performance on any of it may not provide an accurate picture of how it will perform on unseen data. For this reason, it is important to split all of my data at the very beginning into a training set and **hold-out set**, then **perform cross-validation on the training set** to tune my model's hyperparameters. After this, I can **select the best hyperparameters** and use the hold-out set, which has not been used at all, to test how well the model can be expected to perform on a dataset that it has never seen before.

## 3. Let's practice!

You already have all the tools to perform this technique. Your old friend train test split and your new pal GridSearchCV will be particularly handy. Have a crack at it in the interactive exercises and we'll see you in the next chapter!















# Preprocessing data

## 1. Preprocessing data

Welcome to the final chapter of this introductory course on supervised learning with scikit-learn! You have learnt how to implement both classification and regression models, how to measure model performance, and how to tune hyperparameters in order to improve performance. However, all the data that you have used so far has been relatively nice and in a format that allows you to plug and play into scikit-learn from the get-go. With real-world data, this will rarely be the case, and instead you will have to preprocess your data before you can build models. In this chapter, you will learn all about this vital preprocessing step.

## 2. Dealing with categorical features

Say you are dealing with a dataset that has categorical features, such as 'red' or 'blue', or 'male' or 'female'. As these are not numerical values, the scikit-learn API will not accept them and you will have to preprocess these features into the correct format. Our goal is to convert these features so that they are numerical. The way we achieve this by splitting the feature into a number of binary features called '**dummy variables**', one for each category: '0' means the observation was not that category, while '1' means it was.

## 3. Dummy variables

For example, say we are dealing with a car dataset that has a 'origin' feature with three different possible values: 'US', 'Asia', and 'Europe'.

## 4. Dummy variables

We create binary features for each of the origins, as each car is made in exactly one country, each row in the dataset will have a one in exactly one of the three columns and zeros in the other two. Notice that in this case, if a car is not from the US and not from Asia, then implicitly, it is from Europe. That means that we do not actually need three separate features, but only two, so we can

## 5. Dummy variables

delete the 'Europe' column. If we do not do this, we are duplicating information, which might be an issue for some models.

## 6. Dealing with categorical features in Python

There are several ways to create dummy variables in Python. In **scikit-learn**, we can use **OneHotEncoder**. Or we can use **pandas**' **get dummies function**. Here, we will use get dummies.

## 7. Automobile dataset

The target variable here is miles per gallon or mpg. Remember that there is one categorical feature, origin, with three possible values: 'US', 'Asia', and 'Europe'.

## 8. EDA w/ categorical feature

Here is a box plot showing how mpg varies by origin. Let's encode this feature using dummy variables.

## 9. Encoding dummy variables

We import pandas, read in the DataFrame, and then apply the get dummies function. Notice, how pandas creates three new binary features. In the third row, origin USA and origin Europe have zeroes, while origin Asia has a one, indicating that the car is of Asian origin. But if origin USA and origin Europe are zero, then we already know that the car is Asian!

## 10. Encoding dummy variables

So, we drop the origin Asia column. Alternatively, we could have passed the "drop first" option to get dummies. Notice that the new column names have the following structure: original column name, underscore, value name. Once we have created our dummy variables, we can fit models as before.

## 11. Linear regression with dummy variables

Here, for example, we fit the ridge regression model to the data and compute its R-squared.

## 12. Let's practice!

Now it's your turn to practice dealing with categorical features. Enjoy!







# Handling missing data



## 1. Handling missing data

We say that data is missing when there is no value for a given feature in a particular row. This can occur in the real-world for many reasons: there may have been no observation, there may have been a transcription error, or the data may have been corrupted. Whatever the case, we, as data scientists, need to deal with it.

## 2. PIMA Indians dataset

Let's now load the PIMA Indians dataset. It doesn't look like it has any missing values as, according to df dot info, all features have 768 non-null entries. However, missing values can be encoded in a number of different ways, such as by zeroes, or question marks, or negative ones.

## 3. PIMA Indians dataset

Checking out df dot head, it looks as though there are observations where insulin is zero. And triceps, which is the thickness of the skin, is zero. These are not possible and, as we have no indication of the real values, the data is, for all intents and purposes, missing.

## 4. Dropping missing data

Before we go any further, let's make all these entries **'NaN'** using the **replace method** on the relevant columns. So, how do we deal with missing data? One way is to drop all rows containing missing data.

## 5. Dropping missing data

We can do so using the pandas DataFrame method dropna. Checking out the shape of the resulting data frame, though, we see that we now have only approximately half the rows left! We've lost half of our data and this is unacceptable. If only a few rows contain missing values, then it's not so bad, but generally we need a more robust method. It is generally an equally bad idea to remove columns that contain NaNs.

## 6. Imputing missing data

Another option is to **impute missing data.** All imputing means is to make an educated guess as to what the missing values could be. A common strategy is, in any given column with missing values, to compute the mean of all the non-missing entries and to **replace all missing values with the mean**. Let's try this now on our dataset. We import Imputer from sklearn dot preprocessing and instantiate an instance of the Imputer: imp. The keyword argument missing values here specifies that missing values are represented by NaN; strategy specifies that we will use the mean as described above; axis equals 0 specifies that we will impute along columns, a '1' would mean rows. Now, we can fit this imputer to our data using the fit method and then transform our data using the transform method! Due to their ability to transform our data as such, imputers are known as transformers, and any model that can transform data this way, using the transform method, is called a transformer. After transforming the data, we could then fit our supervised learning model to it, but is there a way to do both at once?

## 7. Imputing within a pipeline

There sure is! We can use the scikit-learn pipeline object. We import Pipeline from sklearn dot pipeline and Imputer from sklearn dot preprocessing. We also instantiate **a log reg mode**l. We then build the Pipeline object! We construct a list of steps in the pipeline, where each step is a 2-tuple containing the name you wish to give the relevant step and the estimator. We then pass this list to the Pipeline constructor. We can split our data into training and test sets and

## 8. Imputing within a pipeline

fit the pipeline to the training set and then predict on the test set, as with any other model. For good measure here, we compute accuracy. Note that, in a pipeline, each step but the last must be a transformer and the last must be an estimator, such as, a classifier or a regressor.

## 9. Let's practice!

Now it's your turn to impute missing data and build machine learning pipelines!







# Centering and scaling

**Got It!**

## 1. Centering and scaling

Great work on imputing data and building machine learning pipelines using scikit-learn! Data imputation is one of several important preprocessing steps for machine learning. In this video, will cover another: centering and scaling your data.

## 2. Why scale your data?

To motivate this, let's use df dot describe to check out the ranges of the feature variables in the red wine quality dataset. The features are chemical properties such as acidity, pH, and alcohol content. The target value is good or bad, encoded as '1' and '0', respectively. We see that the ranges vary widely: 'density' varies from (point) 99 to to 1 and 'total sulfur dioxide' from 6 to 289!

## 3. Why scale your data?

Many machine learning models use some form of distance to inform them so **if you have features on far larger scales, they can unduly influence your model.** For example, K-nearest neighbors uses distance explicitly when making predictions. For this reason, we actually want features to be on a similar scale. To achieve this, we do what is called normalizing or scaling and centering.

## 4. Ways to normalize your data

There are several ways to normalize your data: given any column, you can subtract the mean and divide by the variance so that all features are centred around zero and have variance one. This is called **standardization**. You can also subtract the minimum and divide by the range of the data so the normalized dataset has minimum zero and maximum one. You can also normalize so that data ranges from -1 to 1 instead. In this video, I'll show you to to perform standardization. See the scikit-learn docs for how to implement the other approaches.

## 5. Scaling in scikit-learn

To scale our features, we import scale from sklearn dot preprocessing. We then pass the feature data to scale and this returns our scaled data. Looking at the mean and standard deviation of the columns of both the original and scaled data verifies this.

## 6. Scaling in a pipeline

We can also put a scalar in a pipeline object! To do so, we import StandardScaler from sklearn dot reprocessing and build a pipeline object as we did earlier; here we'll use a K-nearest neighbors algorithm. We then split our wine quality dataset in training and test sets, fit the pipeline to our training set, and predict on our test set. Computing the accuracy yields (point) 956, whereas performing KNN without scaling resulted in an accuracy of (point) 928. Scaling did improve our model performance!

## 7. CV and scaling in a pipeline

Let's also take a look at how we can use cross-validation with a supervised learning pipeline. We first build our pipeline. We then specify our hyperparameter space by creating a dictionary: the keys are the pipeline step name followed by a double underscore, followed by the hyperparameter name; the corresponding value is a list or an array of the values to try for that particular hyperparameter. In this case, we are tuning only the n neighbors in the KNN model. As always, we split our data into cross-validation and hold-out sets. We then perform a GridSearch over the parameters in the pipeline by instantiating the GridSearchCV object and fitting it to our training data. The predict method will call predict on the estimator with the best found parameters and we do this on the hold-out set.

## 8. Scaling and CV in a pipeline

We also print the best parameters chosen by our gridsearch, along with the accuracy and classification report of the predictions on the hold-out set.

## 9. Let's practice!






















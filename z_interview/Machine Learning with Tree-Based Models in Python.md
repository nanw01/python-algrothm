# Machine Learning with Tree-Based Models in Python

## Classification and Regression Trees

## 1. Decision-Tree for Classification

Hi! My name is Elie Kawerk, I'm a Data Scientist and I'll be your instructor. In this course, you'll be learning about tree-based models for classification and regression.

## 2. Course Overview

In chapter 1, you'll be introduced to a set of supervised learning models known as **Classification-And-Regression-Tree** or CART. In chapter 2, you'll understand the notions of **bias-variance trade-off** and **model ensembling**. Chapter 3 introduces you to **Bagging and Random Forests.** Chapter 4 deals with boosting, specifically with **AdaBoost and Gradient Boosting**. Finally in chapter 5, you'll understand how to get the most out of your models through **hyperparameter-tuning**.

## 3. Classification-tree

Given a labeled dataset, a classification tree learns a sequence of if-else questions about individual features in order to infer the labels. In contrast to linear models, trees are able to capture non-linear relationships between features and labels. In addition, trees don't require the features to be on the same scale through standardization for example.

## 4. Breast Cancer Dataset in 2D

To understand trees more concretely, we'll try to predict whether a tumor is malignant or benign in the Wisconsin Breast Cancer dataset using only 2 features. The figure here shows a scatterplot of two cancerous cell features with malignant-tumors in blue and benign-tumors in red.

## 5. Decision-tree Diagram

When a classification tree is trained on this dataset, the tree learns a sequence of if-else questions with each question involving one feature and one split-point. Take a look at the tree diagram here. At the top, the tree asks whether the concave-points mean of an instance is <= 0-point-051. If it is, the instance traverses the True **branch**; otherwise, it traverses the False **branch**. Similarly, the instance keeps traversing the internal branches until it reaches an end. The label of the instance is then predicted to be that of the prevailing class at that end. The maximum number of branches separating the top from an extreme-end is known as the maximum depth which is equal to 2 here.

## 6. Classification-tree in scikit-learn

Now that you know what a classification tree is, let's fit one with scikit-learn. First, import DecisionTreeClassifier from sklearn.tree as shown in line 1. Also, import the functions train_test_split() from sklearn.model_selection and accuracy_score() from sklearn.metrics. In order to obtain an unbiased estimate of a model's performance, you must evaluate it on an unseen test set. To do so, first split the data into 80% train and 20% test using train_test_split(). Set the parameter stratify to y in order for the train and test sets to have the same proportion of class labels as the unsplit dataset. You can now use DecisionTreeClassifier() to instantiate a tree classifier, dt with a maximum depth of 2 by setting the parameter max_depth to 2. Note that the parameter random_state is set to 1 for reproducibility.

```python
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
stratify=y,
random_state=1)
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
```

## 7. Classification-tree in scikit-learn

Then call the fit method on dt and pass X_train and y_train. To predict the labels of the test-set, call the predict method on dt. Finally print the accuracy of the test set using **accuracy_score()**. To understand the tree's predictions more concretely, let's see how it classifies instances in the feature-space.

```python
# Fit dt to the training set
dt.fit(X_train,y_train)
# Predict test set labels
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)
```



## 8. Decision Regions

**A classification-model divides the feature-space into regions where all instances in one region are assigned to only one class-label. These regions are known as decision-regions.** **Decision-regions are separated by surfaces called decision-boundaries.** The figure here shows the decision-regions of a linear-classifier. Note how the boundary is a straight-line.

## 9. Decision Regions: CART vs. Linear Model

In contrast, as shown here on the right, a classification-tree produces rectangular decision-regions in the feature-space. This happens because at each split made by the tree, only one feature is involved.

## 10. Let's practice!

Now let's practice!



# Classification tree Learning

## 1. Classification-Tree Learning

Welcome back! In this video, you'll examine how a classification-tree learns from data.

## 2. Building Blocks of a Decision-Tree

Let's first start by defining some terms. **A decision-tree is a data-structure consisting of a hierarchy of individual units called nodes. A node is a point that involves either a question or a prediction.**

## 3. Building Blocks of a Decision-Tree

**The root is the node at which the decision-tree starts growing.** It has **no parent node and involves a question that gives rise to 2 children nodes through two branches**. An **internal node is a node that has a parent. It also involves a question that gives rise to 2 children nodes**. Finally, **a node that has no children is called a leaf.** **A leaf has one parent node and involves no questions. It's where a prediction is made.** Recall that when a classification tree is trained on a labeled dataset, the tree learns patterns from the features in such a way to produce the purest leafs. In other words the tree is trained in such a way so that, in each leaf, one class-label is predominant.

## 4. Prediction

In the tree diagram shown here, consider the case where an instance traverses the tree to reach the leaf on the left. In this leaf, there are 257 instances classified as benign and 7 instances classified as malignant. As a result, the tree's prediction for this instance would be: 'benign'. In order to understand how a classification tree produces the purest leafs possible, let's first define the concept of information gain.

![image-20210315012526912](https://tva1.sinaimg.cn/large/e6c9d24ely1gokiw6zaaej20u40h4ag0.jpg)

## 5. Information Gain (IG)

The nodes of a classification tree are grown recursively; in other words, the obtention of an internal node or a leaf depends on the state of its predecessors. To produce the purest leafs possible, at each node, a tree **asks a question involving one feature f and a split-point sp**. But how does it know which feature and which split-point to pick? It does so by **maximizing Information gain!** The tree considers that every node contains information and **aims at maximizing the Information Gain** obtained after each split. Consider the case where a node with N samples is split into a left-node with Nleft samples and a right-node with Nright samples.

## 6. Information Gain (IG)

The information gain for such split is given by the formula shown here. A question that you may have in mind here is: 'What criterion is used to measure the **impurity** of a node?' Well, there are different criteria you can use among which are the **gini-index** and **entropy**. Now that you know what is Information gain, let's describe how a classification tree learns.

![image-20210315013348438](https://tva1.sinaimg.cn/large/e6c9d24ely1gokj4sk8hqj216i07g401.jpg)

## 7. Classification-Tree Learning

When an unconstrained tree is trained, **the nodes are grown recursively**. In other words, a node exists based on the state of its predecessors. At a non-leaf node, the data is split based on feature f and split-point sp in such a way to maximize information gain. If the information gain obtained by splitting a node is null, the node is declared a leaf. Keep in mind that these rules are for unconstrained trees. If you constrain the maximum depth of a tree to 2 for example, all nodes having a depth of 2 will be declared leafs even if the information gain obtained by splitting such nodes is not null.

![image-20210315013537898](https://tva1.sinaimg.cn/large/e6c9d24ely1gokj6n9l6pj20pe0a40u7.jpg)

## 8. Information Criterion in scikit-learn (Breast Cancer dataset)

Revisiting the 2D breast-cancer dataset from the previous lesson, you can set the information criterion of dt to the gini-index by setting the criterion parameter to 'gini' as shown on the last line here.

## 9. Information Criterion in scikit-learn

Now fit dt to the training set and predict the test set labels. Then determine dt's test set accuracy which evaluates to about 92%.

## 10. Let's practice!

Now it's your turn to practice.



# Decision tree for regression

##  1. Decision-Tree for Regression

Welcome back! In this video, you'll learn how to train a decision tree for a regression problem. Recall that in regression, the target variable is continuous. In other words, the output of your model is a real value.

## 2. Auto-mpg Dataset

Let's motivate our discussion of regression by introducing the automobile miles-per-gallon dataset from the UCI Machine Learning Repository. This dataset consists of 6 features corresponding to the characteristics of a car and a continuous target variable labeled mpg which stands for miles-per-gallon. Our task is to predict the mpg consumption of a car given these six features. To simplify the problem, here the analysis is restricted to only one feature corresponding to the displacement of a car. This feature is denoted by displ.

## 3. Auto-mpg with one feature

A 2D scatter plot of mpg versus displ shows that the mpg-consumption decreases nonlinearly with displacement. Note that linear models such as linear regression would not be able to capture such a non-linear trend. Let's see how you can train a decision tree with scikit-learn to solve this regression problem.

## 4. Regression-Tree in scikit-learn

Note that the features X and the labels y are already loaded in the environment. First, import **DecisionTreeRegressor** from sklearn-dot-tree and the functions train_test_split() from sklearn-dot-model_selection and mean_squared_error as MSE() from sklearn-dot-metrics. Then, split the data into 80%-train and 20%-test using train_test_split. You can now instantiate the **DecisionTreeRegressor**() with a maximum depth of 4 by setting the parameter max_depth to 4. In addition, set the parameter min_sample_leaf to 0-dot-1 to impose a stopping condition in which each leaf has to contain at least 10% of the training data.

## 5. Regression-Tree in scikit-learn

Now fit dt to the training set and predict the test set labels. To obtain the root-mean-squared-error of your model on the test-set; proceed as follows: - first, evaluate the mean-squared error, - then, raise the obtained value to the power 1/2. Finally, print dt's test set rmse to obtain a value of 5-dot-1.

## 6. Information Criterion for Regression-Tree

Here, it's important to note that, when a regression tree is trained on a dataset, the **impurity of a node** is measured using the **mean-squared error** of the targets in that node. This means that the regression tree tries to find the splits that produce leafs where in each leaf the target values are on average, the closest possible to the mean-value of the labels in that particular leaf.

![image-20210315014458778](https://tva1.sinaimg.cn/large/e6c9d24ely1gokjgeuflzj21660e4gon.jpg)

## 7. Prediction

As a new instance traverses the tree and reaches a certain leaf, its target-variable 'y' is computed as the average of the target-variables contained in that leaf as shown in this formula.

  ![image-20210315014703219](https://tva1.sinaimg.cn/large/e6c9d24ely1gokjiiemg9j20n405umxo.jpg)

## 8. Linear Regression vs. Regression-Tree

To highlight the importance of the flexibility of regression trees, take a look at this figure. On the left we have a scatter plot of the data in blue along with the predictions of a linear regression model shown in black. **The linear model fails to capture the non-linear trend exhibited by the data.** On the right, we have the same scatter plot along with a red line corresponding to the predictions of the regression tree that you trained earlier. **The regression tree shows a greater flexibility and is able to capture the non-linearity, though not fully.** In the next chapter, you'll aggregate the predictions of a set of trees that are trained differently to obtain better results.

## 9. Let's practice!

Now it's your turn to practice.





## The Bias-Variance Tradeoff

# Generalization Error

##  1. Generalization Error

Welcome to chapter 2! In this video, you'll understand what is the generalization error of a supervised machine learning model.

## 2. Supervised Learning - Under the Hood

In supervised learning, you make the assumption that there's a mapping f between features and labels. You can express this as y=f(x). f which is shown in red here is an unknown function that you want to determine. In reality, data generation is always accompanied with randomness or noise like the blue points shown here.

## 3. Goals of Supervised Learning

Your goal is to find a model fhat that best approximates f. When training fhat, you want to make sure that noise is discarded as much as possible. At the end, fhat should achieve a low predictive error on unseen datasets.

## 4. Difficulties in Approximating $f$

You may encounter two difficulties when approximating f. The first is **overfitting**, it's when fhat fits the noise in the training set. The second is **underfitting**, it's when fhat is not flexible enough to approximate f.

## 5. Overfitting

**When a model overfits the training set, its predictive power on unseen datasets is pretty low.** This is illustrated by the predictions of the decision tree regressor shown here in red. The model clearly memorized the noise present in the training set. Such model achieves a low training set error and a high test set error.

## 6. Underfitting

**When a model underfits the data**, like the regression tree whose predictions are shown here in red, **the training set error is roughly equal to the test set error.** However, **both errors are relatively high**. Now the trained model isn't flexible enough to capture the complex dependency between features and labels. In analogy, it's like teaching calculus to a 3-year old. The child does not have the required mental abstraction level that enables him to understand calculus.

## 7. Generalization Error

**The generalization error of a model tells you how much it generalizes on unseen data**. I can be decomposed into 3 terms: **bias**, **variance** and **irreducible error** where the irreducible error is the error contribution of noise.

![image-20210315015944951](https://tva1.sinaimg.cn/large/e6c9d24ely1gokjvqb0orj20pc034mxf.jpg)

## 8. Bias

**The bias term tells you, on average, how much fhat and f are different**. To illustrate this consider the high bias model shown here in black; this model is not flexible enough to approximate the true function f shown in red. High bias models lead to underfitting.

## 9. Variance

**The variance term tells you how much fhat is inconsistent over different training sets.** Consider the high variance model shown here in black; in this case, fhat follows the training data points so closely that it misses the true function f shown in red. High variance models lead to overfitting.

## 10. Model Complexity

**The complexity of a model sets its flexibility to approximate the true function f.** For example: increasing the maximum-tree-depth increases the complexity of a decision tree.

## 11. Bias-Variance Tradeoff

The diagram here shows how the best model complexity corresponds to the lowest generalization error. **When the model complexity increases, the variance increases while the bias decreases.** Conversely, **when model complexity decreases, variance decreases and bias increases.** **Your goal is to find the model complexity that achieves the lowest generalization error.** Since this error is the sum of three terms with **the irreducible error being constant**, you **need to find a balance between bias and variance because as one increases the other decreases. This is known as the bias-variance trade-off.**

## 12. Bias-Variance Tradeoff: A Visual Explanation

Visually, you can imagine approximating fhat as aiming at the center of a shooting-target where the center is the true function f. If fhat is low bias and low variance, your shots will be closely clustered around the center. If fhat is high variance and high bias, not only will your shots miss the target but they would also be spread all around the shooting target.

![image-20210315020605298](https://tva1.sinaimg.cn/large/e6c9d24ely1gokk2byaktj20fm0eu76h.jpg)

## 13. Let's practice!

Time to put this into practice.





# Diagnose bias and variance problems

## 1. Diagnosing Bias and Variance Problems

In this video, you'll learn how to diagnose bias and variance problems.

## 2. Estimating the Generalization Error

Given that you've trained a supervised machine learning model labeled fhat, how do you **estimate the fhat's generalization error**? This **cannot be done directly** because: - **f is unknown**, - usually you only have **one dataset**, - you don't have access to the error term due to noise.(**noise is unpredictable **)

## 3. Estimating the Generalization Error

A solution to this is **to first split the data into a training and test set**. The **model fhat can then be fit to the training set** and its **error can be evaluated on the test set**. The generalization error of fhat is r**oughly approximated by fhat's error on the test set**.

## 4. Better Model Evaluation with Cross-Validation

Usually, the test set should be kept untouched until one is confident about fhat's performance. It should only be used to evaluate fhat's final performance or error. Now, evaluating fhat's performance on the training set may produce an optimistic estimation of the error because fhat was already exposed to the training set when it was fit. **To obtain a reliable estimate of fhat's performance, you should use a technique called cross-validation or CV.** CV can be performed using K-Fold-CV or hold-out-CV . In this lesson, we'll only be explaining K-fold-CV.

## 5. K-Fold CV

The diagram here illustrates this technique for K=10: - First, the training set (T) is split randomly into 10 partitions or folds, - The error of fhat is evaluated 10 times on the 10 folds, - Each time, one fold is picked for evaluation after training fhat on the other 9 folds. - At the end, you'll obtain a list of 10 errors.

![image-20210315123019441](https://tva1.sinaimg.cn/large/e6c9d24ely1gol23toir2j21dz0cr0to.jpg)

## 6. K-Fold CV

Finally, as shown in this formula, the CV-error is computed as the mean of the 10 obtained errors.

## 7. Diagnose Variance Problems

Once you have **computed fhat's cross-validation-error, you can check if it is greater than fhat's training set error.** **If it is greater, fhat is said to suffer from high variance**. In such case, fhat has overfit the training set. To remedy this try decreasing fhat's complexity. For example, in a decision tree you can reduce the maximum-tree-depth or increase the maximum-samples-per-leaf. In addition, you may also gather more data to train fhat.

![image-20210315123515644](https://tva1.sinaimg.cn/large/e6c9d24ely1gol28xzjfbj21280cswg6.jpg)

## 8. Diagnose Bias Problems

On the other hand, fhat is said to suffer from high bias if its cross-validation-error is roughly equal to the training error but much greater than the desired error. In such case fhat underfits the training set. To remedy this try increasing the model's complexity or gather more relevant features for the problem.

![image-20210315123533061](https://tva1.sinaimg.cn/large/e6c9d24ely1gol2996n2hj219s0cqq4s.jpg)

## 9. K-Fold CV in sklearn on the Auto Dataset

Let's now see how we can perform K-fold-cross-validation using scikit-learn on the auto-dataset which is already loaded. In addition to the usual imports, you should also import the function cross_val_score() from sklearn-dot-model_selection. First, split the dataset into 70%-train and 30%-test using **train_test_split()**. Then, instantiate a DecisionTreeRegressor() dt with the parameters max_depth set to 4 and min_samples_leaf to 0-dot-14.

## 10. K-Fold CV in sklearn on the Auto Dataset

Next, call cross_val_score() by passing dt, X_train, y_train; set the parameters cv to 10 for 10-fold-cross-validation and scoring to neg_mean_squared_error to compute the negative-mean-squared-errors. The scoring parameter was set so because cross_val_score() does not allow computing the mean-squared-errors directly. Finally, set n_jobs to -1 to exploit all available CPUs in computation. The result is a numpy-array of the 10 negative mean-squared-errors achieved on the 10-folds. You can multiply the result by minus-one to obtain an array of CV-MSE. After that, fit dt to the training set and evaluate the labels of the training and test sets.

## 11. K-Fold CV in sklearn on the Auto Dataset

The CV-mean-squared-error can be determined as the mean of MSE_CV. Finally, you can use the function MSE to evaluate the train and test set mean-squared-errors. Given that the training set error is smaller than the CV-error, we can deduce that dt overfits the training set and that it suffers from high variance. Notice how the CV and test set errors are roughly equal.

## 12. Let's practice!

Now it's your turn.



# Ensemble Learning

## 1. Ensemble Learning

In this lesson, you will learn about **a supervised learning technique** known as **ensemble learning**.

## 2. Advantages of CARTs

Let's first recap what we learned from the previous chapter about CARTs(Classification And Regression Tree). CARTs present many advantages. For example they are **easy to understand** and **their output is easy to interpret**. In addition, **CARTs are easy to use** and their **flexibility gives them an ability to describe nonlinear dependencies** between features and labels. Moreover, you don't need a lot of feature preprocessing to train a CART. In contrast to other models, you **don't have to standardize or normalize features** before feeding them to a CART.

## 3. Limitations of CARTs

CARTs also have limitations. A classification tree for example, is **only able to produce orthogonal decision boundaries**. CARTs are also **very sensitive to small variations in the training set**. Sometimes, when a single point is removed from the training set, a CART's learned parameters may changed drastically. CARTs also suffer from **high variance** **when** they are trained **without constraint**s. In such case, they may overfit the training set. A **solution** that takes advantage of the flexibility of CARTs while reducing their tendency to memorize noise is **ensemble learning**.

## 4. Ensemble Learning

Ensemble learning can be summarized as follows: -As a first step, **different models are trained on the same dataset**. -**Each model makes its own predictions.** -A **meta-model** then aggregates the predictions of individual models and outputs a final prediction. -The final prediction is **more robust and less prone to errors** than each individual model. -The best results are obtained when the models are skillful but in different ways meaning that if some models make predictions that are way off, the other models should compensate these errors. In such case, the meta-model's predictions are more robust.

## 5. Ensemble Learning: A Visual Explanation

Let's take a look at the diagram here to visually understand how ensemble learning works for a classification problem. First, the training set is fed to different classifiers. Each classifier learns its parameters and makes predictions. Then these predictions are fed to a meta model which aggregates them and outputs a final prediction.

![image-20210315180332278](https://i.loli.net/2021/03/16/rMgf5EjtlOXIivp.png)

## 6. Ensemble Learning in Practice: Voting Classifier

Let's now take a look at an ensemble technique known as the **voting classifier.** More concretely, we'll consider a **binary classification** task. The ensemble here consists of N classifiers making the predictions P0,P1,to,PN with P=0-or-1. **The meta model outputs the final prediction by hard voting.**

## 7. Hard Voting

To understand hard voting, consider a voting classifier that consists of 3 trained classifiers as shown in the diagram here. While classifiers 1 and 3 predict the label of 1 for a new data-point, classifier 2 predicts the label 0. In this case, 1 has 2 votes while 0 has 1 vote. As a result, the voting classifier predicts 1.

![image-20210315180629512](https://i.loli.net/2021/03/16/M7EbHuzOfZLvgeB.png)

## 8. Voting Classifier in sklearn (Breast-Cancer dataset)

Now that you know what a voting classifier is, let's train one on the breast cancer dataset using scikit-learn. You'll do so using all the features in the dataset to predict whether a cell is malignant or not. In addition to the usual imports, import LogisticRegression, DecisionTreeClassifier and KNeighborsClassifier. You also need to import VotingClassifier from sklearn-dot-ensemble.

```python
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_
score
from sklearn.model
_
selection import train
_
test
_
split
# Import models, including VotingClassifier meta-model
from sklearn.linear
_
model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
# Set seed for reproducibility
SEED = 1
```



## 9. Voting Classifier in sklearn (Breast-Cancer dataset)

Then, split the data into 70%-train and 30%-test and instantiate the different models as shown here. After that, define a list named classifiers that contains tuples corresponding the the name of the models and the models themselves.、

```python
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3,random_state= SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),('K Nearest Neighbours', knn),('Classification Tree', dt)]
```



## 10. Voting Classifier in sklearn (Breast-Cancer dataset)

You can now write a for loop to iterate over the list classifiers; fit each classifier to the training set, evaluate its accuracy on the test set and print the result. The output shows that the best classifier LogisticRegression achieves an accuracy of 94-dot-7%.

```python
# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
#fit clf to the training set
clf.fit(X_train, y_train)
# Predict the labels of the test set
y_pred = clf.predict(X_test)
# Evaluate the accuracy of clf on the test set
print('{:s} : {:.3f}'
.format(clf_name, accuracy_score(y_test, y_pred)))
```



## 11. Voting Classifier in sklearn (Breast-Cancer dataset)

Finally, you can instantiate a voting classifier vc by setting the estimators parameter to classifiers. Fitting vc to the training set yields a test set accuracy of 95-dot-3%. This accuracy is higher than that achieved by any of the individual models in the ensemble.

```python
# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)
# Fit 'vc' to the traing set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)
# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {.3f}'.format(accuracy_score(y_test, y_pred)))
```



## 12. Let's practice!

Now it's time to put this into practice.



#### soft voting and hard voting





# Bagging and Random Forests

# Bagging

## 1. Bagging

Welcome back! In this video, you'll be introduced to an ensemble method known as Bootstrap aggregation or Bagging.

## 2. Ensemble Methods

In the last chapter, you learned that the **Voting Classifier** is an ensemble of models that are fit to the same training set using different algorithms. You also saw that the final predictions were obtained by majority voting. **In Bagging, the ensemble is formed by models that use the same training algorithm.** However, these models are not trained on the entire training set. Instead, each model is trained on a different subset of the data.

## 3. Bagging

In fact, bagging stands for **bootstrap aggregation**. Its name refers to the fact that it uses a technique known as the bootstrap. Overall, Bagging has the effect of reducing the variance of individual models in the ensemble.

## 4. Bootstrap

Let's first try to understand what the bootstrap method is. Consider the case where you have 3 balls labeled A, B, and C. **A bootstrap sample is a sample drawn from this with replacement.** By replacement, we mean that any ball can be drawn many times. For example, in the first bootstrap sample shown in the diagram here, B was drawn 3 times in a raw. In the second bootstrap sample, A was drawn two times while B was drawn once, and so on. You may now ask how bootstraping can help us produce an ensemble.

![image-20210316111146464](https://i.loli.net/2021/03/16/eu2qfnRV8tmDg7T.png)

## 5. Bagging: Training

In fact, in the training phase, bagging consists of drawing N different bootstrap samples from the training set. As shown in the diagram here, each of these bootstrap samples are then used to train N models that use the same algorithm .

## 6. Bagging: Prediction

When a new instance is fed to the different models forming the bagging ensemble, each model outputs its prediction. The meta model collects these predictions and outputs a final prediction depending on the nature of the problem.

## 7. Bagging: Classification & Regression

In classification, the final prediction is obtained by majority voting. The corresponding classifier in scikit-learn is **BaggingClassifier**. In regression, the final prediction is the average of the predictions made by the individual models forming the ensemble. The corresponding regressor **in scikit-learn** is **BaggingRegressor**.

## 8. Bagging Classifier in sklearn (Breast-Cancer dataset)

Great! Now that you understand how Bagging works, let's train a **BaggingClassifier** in scikit-learn on the breast cancer dataset. Note that the dataset is already loaded. First import **BaggingClassifier**, **DecisionTreeClassifier**, **accuracy_score** and **train_test_split** and then split the data into 70%-train and 30%-test as shown here.

## 9. Bagging Classifier in sklearn (Breast-Cancer dataset)

Now, instantiate a classification tree dt with the parameters max_depth set to 4 and min_samples_leaf set to 0-dot-16. You can then instantiate a BaggingClassifier bc that consists of 300 classification trees dt. This can be done by setting the parameters base_estimator to dt and n_estimators to 300. In addition, set the paramter n_jobs to -1 so that all CPU cores are used in computation. Once you are done, fit bc to the training set, predict the test set labels and finally, evaluate the test set accuracy. The output shows that a BaggingClassifier achieves a test set accuracy of 93-dot-6%. Training the classification tree dt, which is the base estimator here, to the same training set would lead to a test set accuracy of 88-dot-9%. The result highlights how bagging outperforms the base estimator dt.



```python
# Import models and utility functions
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
stratify=y,
random_state=SEED)

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)
# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))

```





## 10. Let's practice!

Alright, now it's your time to practice.



# Out of Bag Evaluation

**Got It!**

## 1. Out Of Bag Evaluation

You will now learn about Out-of-bag evaluation.

## 2. Bagging

Recall that in bagging, some instances may be sampled several times for one model. On the other hand, other instance may not be sampled at all.

## 3. Out Of Bag (OOB) instances

On average, for each model, 63% of the training instances are sampled. **The remaining 37% that are not sampled constitute what is known as the Out-of-bag or OOB instances.** Since OOB instances are not seen by a model during training, these can be used to estimate the performance of the ensemble without the need for cross-validation. This technique is known as OOB-evaluation.

## 4. OOB Evaluation

To understand OOB-evaluation more concretely, take a look at this diagram. Here, for each model, the bootstrap instances are shown in blue while the OOB-instances are shown in red. Each of the N models constituting the ensemble is then trained on its corresponding bootstrap samples and evaluated on the OOB instances. This leads to the obtainment of N OOB scores labeled OOB1 to OOBN. The OOB-score of the bagging ensemble is evaluated as the average of these N OOB scores as shown by the formula on top.

![image-20210316120130707](https://i.loli.net/2021/03/17/JylOQ8ULHz3Tk4F.png)

## 5. OOB Evaluation in sklearn (Breast Cancer Dataset)

Alright! Now it's time to see OOB-evaluation in action. Again, we'll be classifying cancerous cells as malignant or benign from the breast cancer dataset which is already loaded. After importing **BaggingClassifier**, **DecisionTreeClassifier**, **accuracy_score** and **train_test_split**, split the dataset in a stratified way into 70%-train and 30%-test by setting the parameter stratify to y.

```python
# Import models and split utility function
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,
stratify= y,
random_state=SEED)
```



## 6. OOB Evaluation in sklearn (Breast Cancer Dataset)

Now, first instantiate a classification tree dt with a maximum-depth of 4 and a minimum percentage of samples per leaf equal to 16%. Then instantiate a BaggingClassifier called bc that consists of 300 classification trees. This can be done by setting the parameters n_estimators to 300 and base_estimator to dt. Importantly, set the parameter oob_score to True in order to evaluate the OOB-accuracy of bc after training. Note that in scikit-learn, the OOB-score corresponds to the accuracy for classifiers and the r-squared score for regressors. Now fit bc to the training set and predict the test set labels.

```python
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)
# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))
```



## 7. OOB Evaluation in sklearn (Breast Cancer Dataset)

Assign the test set accuracy to test_accuracy. Finally, evaluate the OOB-accuracy of bc by extracting the attribute oob_score_ from the trained instance; assign the result to oob_accuracy and print out the results. The test-set accuracy is about 93.6% and the OOB-accuracy is about 92.5%. The two obtained accuracies are pretty close though not exactly equal. These results highlight how OOB-evaluation can be an efficient technique to obtain a performance estimate of a bagged-ensemble on unseen data without performing cross-validation.

## 8. Let's practice!

Now let's try some examples.





# Random Forests (RF)

## 1. Random Forests

You will now learn about another ensemble learning method known as **Random Forests**.

## 2. Bagging

Recall that **in bagging the base estimator could be any model including a decision tree, logistic regression or even a neural network.** Each estimator is trained on a distinct bootstrap sample drawn from the training set using all available features.

## 3. Further Diversity with Random Forests

**Random Forests is an ensemble method that uses a decision tree as a base estimator.** In Random Forests, each estimator is trained on a different **bootstrap sample** having the same size as the training set. Random forests introduces further randomization than bagging when training each of the base estimators. When each tree is trained, only d features can be sampled at each node without replacement, where d is a number smaller than the total number of features.

## 4. Random Forests: Training

The diagram here shows the training procedure for random forests. Notice how each tree forming the ensemble is trained on a different bootstrap sample from the training set. In addition, when a tree is trained, at each node, only d features are sampled from all features without replacement. The node is then split using the sampled feature that maximizes information gain. In scikit-learn **d defaults to the square-root of the number of feature**s. For example, if there are 100 features, only 10 features are sampled at each node.

![image-20210316145503480](https://i.loli.net/2021/03/17/Jl4DKwtBH62VG3W.png)

## 5. Random Forests: Prediction

Once trained, predictions can be made on new instances. When a new instance is fed to the different base estimators, each of them outputs a prediction. The predictions are then collected by the random forests meta-classifier and a final prediction is made depending on the nature of the problem.

![image-20210316145651126](https://i.loli.net/2021/03/17/YQZ7F5SEgKmy81U.png)

## 6. Random Forests: Classification & Regression

For classification, the final prediction is made by majority voting. The corresponding scikit-learn class is **RandomForestClassifier**. For regression, the final prediction is the average of all the labels predicted by the base estimators. The corresponding scikit-learn class is **RandomForestRegressor**. In general, Random Forests achieves a lower variance than individual trees.

## 7. Random Forests Regressor in sklearn (auto dataset)

Alright, now it's time to put all this into practice. Here, you'll train a random forests regressor to the auto-dataset which you were introduced to in previous chapters. Note that the dataset is already loaded. After importing RandomForestRegressor, train_test_split and mean_squared_error as MSE, split the dataset into 70%-train and 30%-test as shown here.

```python
# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3,
random_state=SEED)
```



## 8. Random Forests Regressor in sklearn (auto dataset)

Then instantiate a RandomForestRegressor consisting of 400 regression trees. This can be done by setting n_estimators to 400. In addition, set min_samples_leaf to 0-dot-12 so that each leaf contains at least 12% of the data used in training. You can now fit rf to the training set and predict the test set labels. Finally, print the test set RMSE. The result shows that rf achieves a test set RMSE of 3-dot-98; this error is smaller than that achieved by a single regression tree which is 4-dot-43.

```python
# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,
min_samples_leaf=0.12,
random_state=SEED)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
```

```python
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```



## 9. Feature Importance

When a tree based method is trained, the predictive power of a feature or its importance can be assessed. In scikit-learn, feature importance is assessed by measuring how much the tree nodes use a particular feature to reduce impurity. Note that the importance of a feature is expressed as a percentage indicating the weight of that feature in training and prediction. Once you train a tree-based model in scikit-learn, the features importances can be accessed by extracting the feature_importance_ attribute from the model.

## 10. Feature Importance in sklearn

To visualize the importance of features as assessed by rf, you can create a pandas series of the features importances as shown here and then sort this series and make a horiztonal-barplot.

```python
import pandas as pd
import matplotlib.pyplot as plt
# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()
```



## 11. Feature Importance in sklearn

The results show that, according to rf, displ, size, weight and hp are the most predictive features.

## 12. Let's practice!

Now let's try some examples.



# Adaboost

**Got It!**

## 1. AdaBoost

**Boosting refers to an ensemble method in which many predictors are trained and each predictor learns from the errors of its predecessor.**

## 2. Boosting

More formally, in boosting many weak learners are combined to form a strong learner. **A weak learner is a model doing slightly better than random guessing.** For example, a decision tree with a maximum-depth of one, known as a decision-stump, is a weak learner.

## 3. Boosting

In boosting, an ensemble of predictors are trained **sequentially** and each predictor tries to correct the errors made by its predecessor. The two boosting methods you'll explore in this course are **AdaBoost** and **Gradient Boosting**.

## 4. Adaboost

AdaBoost stands for **Adaptive Boosting**. In AdaBoost, **each predictor pays more attention to the instances wrongly predicted by its predecessor** by constantly **changing the weights of training instances**. Furthermore, each predictor is assigned a coefficient alpha that weighs its contribution in the ensemble's final prediction. Alpha depends on the predictor's training error.

## 5. AdaBoost: Training

As shown in the diagram, there are N predictors in total. First, predictor1 is trained on the initial dataset (X,y), and the training error for predictor1 is determined. This error can then be used to determine alpha1 which is predictor1's coefficient. Alpha1 is then used to determine the weights W(2) of the training instances for predictor2. Notice how the incorrectly predicted instances shown in green acquire higher weights. When the weighted instances are used to train predictor2, this predictor is forced to pay more attention to the incorrectly predicted instances. This process is repeated sequentially, until the N predictors forming the ensemble are trained.

![image-20210316151025578](https://i.loli.net/2021/03/17/wxutmQycKkXWNHU.png)

## 6. Learning Rate

An important paramter used in training is the **learning rate**, eta. **Eta is a number between 0 and 1; it is used to shrink the coefficient alpha of a trained predictor.** It's important to note that there's **a tradeoff between eta and the number of estimators**. A smaller value of eta should be compensated by a greater number of estimators.

![image-20210316151850636](https://i.loli.net/2021/03/17/wkHFxXZJO4urhUn.png)

## 7. AdaBoost: Prediction

Once all the predictors in the ensemble are trained, the label of a new instance can be predicted depending on the nature of the problem. For classification, each predictor predicts the label of the new instance and the ensemble's prediction is obtained by weighted majority voting. For regression, the same procedure is applied and the ensemble's prediction is obtained by performing a weighted average. It's important to note that individual predictors need not to be CARTs. However CARTs are used most of the time in boosting because of their high variance.

## 8. AdaBoost Classification in sklearn (Breast Cancer dataset)

Alright, let's fit an AdaBoostClassifier to the breast cancer dataset and evaluate its ROC-AUC score. Note that the dataset is already loaded. After importing AdaBoostClassifier, DecisionTreeClassifier, roc_auc_score, and train_test_split, split the data into 70%-train and 30%-test as shown here.

```python
# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Set seed for reproducibility
SEED = 1
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
stratify=y,
random_state=SEED)
```

## 9. AdaBoost Classification in sklearn (Breast Cancer dataset)

Now instantiate a DecisionTreeClassifier with the parameter max_depth set to 1. After that, instantiate an AdaBoostClassifier called adb_clf consisting of 100 decision-stumps. This can be done by setting the parameters base_estimator to dt and n_estimators to 100. Then, fit adb_clf to the training set and predict the probability of obtaining the positive class in the test set as shown here. This enables you to evaluate the ROC-AUC score of adb_clf by calling the function roc_auc_score and passing the parameters y_test and y_pred_proba.

```python
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)
```



## 10. AdaBoost Classification in sklearn (Breast Cancer dataset)

Finally, you can print the result which shows that the **AdaBoostClassifier** achieves a ROC-AUC score of about 0-dot-99.

```python
# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
```



## 11. Let's practice!

Now it's your turn.







# Gradient Boosting (GB)

## 1. Gradient Boosting (GB)

Gradient Boosting is a popular boosting algorithm that has a proven track record of winning many machine learning competitions.

## 2. Gradient Boosted Trees

In gradient boosting, **each predictor in the ensemble corrects its predecessor's error**. **In contrast to AdaBoost, the weights of the training instances are not tweaked.** Instead, each predictor is trained using the residual errors of its predecessor as labels. In the following slides, you'll explore the technique known as gradient boosted trees where the base learner is a CART.

## 3. Gradient Boosted Trees for Regression: Training

To understand how gradient boosted trees are trained for a regression problem, take a look at the diagram here. The ensemble consists of N trees. Tree1 is trained using the features matrix X and the dataset labels y. The predictions labeled y1hat are used to determine the training set residual errors r1. Tree2 is then trained using the features matrix X and the residual errors r1 of Tree1 as labels. The predicted residuals r1hat are then used to determine the residuals of residuals which are labeled r2. This process is repeated until all of the N trees forming the ensemble are trained.

![image-20210316153105631](https://i.loli.net/2021/03/17/OfEhsSrLT7yZjxe.png)

## 4. Shrinkage

**An important parameter used in training gradient boosted trees is shrinkage.** In this context, shrinkage refers to the fact that the prediction of each tree in the ensemble is shrinked after it is multiplied by a **learning rate eta** which is a number **between 0 and 1.** Similarly to AdaBoost, there's a trade-off between eta and the number of estimators. **Decreasing the learning rate needs to be compensated by increasing the number of estimators in order for the ensemble to reach a certain performance.**

## 5. Gradient Boosted Trees: Prediction

Once all trees in the ensemble are trained, prediction can be made. When a new instance is available, each tree predicts a label and the final ensemble prediction is given by the formula shown on this slide. In scikit-learn, the class for a gradient boosting regressor is **GradientBoostingRegressor**. Though not discussed in this course, a similar algorithm is used for classification problems. The class implementing gradient-boosted-classification in scikit-learn is **GradientBoostingClassifier**.

## 6. Gradient Boosting in sklearn (auto dataset)

Great! Now it's time to get your hands dirty by predicting the miles per gallon consumption of cars in the auto-dataset. Note that the dataset is already loaded. First, import GradientBoostingRegressor from sklearn.ensemble. Also, import the functions train_test_split and mean_squared_error as MSE as shown here. Then split the dataset into 70%-train and 30%-test.

```python
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.3,
random_state=SEED)
```

## 7. Gradient Boosting in sklearn (auto dataset)

Now instantiate a GradientBoostingRegressor gbt consisting of 300 decision-stumps. This can be done by setting the parameters n_estimators to 300 and max_depth to 1. Finally, fit gbt to the training set and predict the test set labels. Compute the test set RMSE as shown here and print the value. The result shows that gbt achieves a test set RMSE of 4-dot-01.

```python
# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)
# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)
# Predict the test set labels
y_pred = gbt.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))
```



## 8. Let's practice!

Time to put this into practice.





# Stochastic Gradient Boosting (SGB)

## 1. Stochastic Gradient Boosting (SGB)



## 2. Gradient Boosting: Cons

**Gradient boosting involves an exhaustive search procedure.** **Each tree in the ensemble is trained to find the best split-points and the best features**. **This procedure may lead to CARTs that use the same split-points and possibly the same features.**

## 3. Stochastic Gradient Boosting

To mitigate these effects, you can use an algorithm known as **stochastic gradient boosting.** In stochastic gradient boosting, each CART is **trained on a random subset of the training data**. **This subset is sampled without replacement.** Furthermore, at the level of each node, features are **sampled without replacement when choosing the best split-points**. As a result, this creates further diversity in the ensemble and the net effect is adding more variance to the ensemble of trees.

## 4. Stochastic Gradient Boosting: Training

Let's take a closer look at the training procedure used in stochastic gradient boosting by examining the diagram shown on this slide. First, instead of providing all the training instances to a tree, only a fraction of these instances are provided through sampling without replacement. The sampled data is then used for training a tree. However, not all features are considered when a split is made. Instead, only a certain randomly sampled fraction of these features are used for this purpose. Once a tree is trained, predictions are made and the residual errors can be computed. These residual errors are multiplied by the learning rate eta and are fed to the next tree in the ensemble. This procedure is repeated sequentially until all the trees in the ensemble are trained. The prediction procedure for a new instance in stochastic gradient boosting is similar to that of gradient boosting.

![image-20210316155526594](https://i.loli.net/2021/03/17/e2g3rRH5yCJuP1q.png)

## 5. Stochastic Gradient Boosting in sklearn (auto dataset)

Alright, now it's time to put this into practice. As in the last video, we'll be dealing with the auto-dataset which is already loaded. Perform the same imports that were introduced in the previous lesson and split the data.

```python
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,
test_size=0.3,
random_state=SEED)
```

## 6. Stochastic Gradient Boosting in sklearn (auto dataset)

Now define a stochastic-gradient-boosting-regressor named sgbt consisting of 300 decision-stumps. This can be done by setting the parameters max_depth to 1 and n_estimators to 300. Here, the parameter subsample was set to 0-dot-8 in order for each tree to sample 80% of the data for training. Finally, the parameter max_features was set to 0-dot-2 so that each tree uses 20% of available features to perform the best-split. Once done, fit sgbt to the training set and predict the test set labels.

```python
# Instantiate a stochastic GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1,
subsample=0.8,
max_features=0.2,
n_estimators=300,
random_state=SEED)
# Fit 'sgbt' to the training set
sgbt.fit(X_train, y_train)
# Predict the test set labels
y_pred = sgbt.predict(X_test)
```



## 7. Stochastic Gradient Boosting in sklearn (auto dataset)

Finally, compute the test set RMSE and print it. The result shows that sgbt achieves a test set RMSE of 3-dot-95.

```python
# Evaluate test set RMSE 'rmse_test'
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print 'rmse_test'
print('Test set RMSE: {:.2f}'.format(rmse_test))
```



## 8. Let's practice!

Now let's try some examples.



# Tuning a CART's Hyperparameters

**Got It!**

## 1. Tuning a CART's hyperparameters

To obtain a better performance, the hyperparameters of a machine learning should be tuned.

## 2. Hyperparameters

Machine learning models are characterized by parameters and hyperparameters. **Parameters are learned from data through training**; examples of parameters include the split-feature and the split-point of a node in a CART. **Hyperparameters are not learned from data; they should be set prior to training**. Examples of hyperparameters include the maximum-depth and the splitting-criterion of a CART.

## 3. What is hyperparameter tuning?

Hyperparameter tuning consists of **searching for the set of optimal hyperparameters for the learning algorithm**. The solution involves **finding the set of optimal hyperparameters yielding an optimal model**. The optimal model **yields an optimal score.** The **score** function measures the agreement between true labels and a model's predictions. In sklearn, it defaults to **accuracy for classifiers and r-squared for regressors**. A model's **generalization performance is evaluated using cross-validation.**

## 4. Why tune hyperparameters?

A legitimate question that you may ask is: why bother tuning hyperparameters? Well, in scikit-learn, a model's **default hyperparameters are not optimal for all problems**. Hyperparameters should be tuned **to obtain the best model performance.**

## 5. Approaches to hyperparameter tuning

Now there are many approaches for hyperparameter tuning including: **grid-search**, **random-search**, and so on. In this course, we'll only be exploring the method of grid-search.

## 6. Grid search cross validation

In grid-search cross-validation, first you **manually set a grid of discrete hyperparameter values**. Then, you **pick a metric for scoring model performance** and you **search exhaustively through the grid**. **For each set of hyperparameters, you evaluate each model's score.** The **optimal hyperparameters are those for which the model achieves the best cross-validation score**. Note that grid-search suffers from the curse of dimensionality, i-dot-e-dot, the bigger the grid, the longer it takes to find the solution.

## 7. Grid search cross validation: example

Let's walk through a concrete example to understand this procedure. Consider the case of a CART where you search through the two-dimensional hyperparameter grid shown here. The dimensions correspond to the CART's maximum-depth and the minimum-percentage of samples per leaf. For each combination of hyperparameters, the cross-validation score is evaluated using k-fold CV for example. Finally, the optimal hyperparameters correspond to the model achieving the best cross-validation score.

## 8. Inspecting the hyperparameters of a CART in sklearn

Let's now see how we can inspect the hyperparameters of a CART in scikit-learn. You can first instantiate a **DecisionTreeClassifier dt** as shown here.

## 9. Inspecting the hyperparameters of a CART in sklearn

Then, call dt's -dot-get_params() method. This prints out a dictionary where the keys are the hyperparameter names. In the following, we'll only be optimizing max_depth, max_features and min_samples_leaf. Note that max_features is the number of features to consider when looking for the best split. When it's a float, it is interpreted as a percentage. You can learn more about these hyperparameters by consulting scikit-learn's documentation.

## 10. Grid search CV in sklearn (Breast Cancer dataset)

Let's now tune dt on the wisconsin breast cancer dataset which is already loaded and split into 80%-train and 20%-test. First, import GridSearchCV from sklearn-dot-model_selection. Then, define a dictionary called params_dt containing the names of the hyperparameters to tune as keys and lists of hyperparameter-values as values. Once done, instantiate a GridSearchCV object grid_dt by passing dt as an estimator and params_dt as param_grid. Also set scoring to accuracy and cv to 10 in order to use 10-fold stratified cross-validation for model evaluation. Finally, fit grid_dt to the training set.

## 11. Extracting the best hyperparameters

After training grid_dt, the best set of hyperparameter-values can be extracted from the attribute -dot-best_params_ of grid_dt. Also, the best cross validation accuracy can be accessed through grid_dt's -dot-best_score_ attribute.

## 12. Extracting the best estimator

Similarly, the best-model can be extracted using the -dot-best_estimator attribute. Note that this model is fitted on the whole training set because the refit parameter of GridSearchCV is set to True by default. Finally, you can evaluate this model's test set accuracy using the score method. The result is about 94-dot-7% while the score of an untuned CART is of 93%.

## 13. Let's practice!

Now it's your turn to practice.







# Tuning a RF's Hyperparameters

## 1. Tuning an RF's Hyperparameters

Let's now turn to a case where we tune the hyperparameters of Random Forests which is an ensemble method.

## 2. Random Forests Hyperparameters

In addition to the hyperparameters of the CARTs forming random forests, the ensemble itself is characterized by other hyperparameters such as **the number of estimators**, whether it uses **bootstraping** or not and so on.

## 3. Tuning is expensive

As a note, hyperparameter tuning is **computationally expensive** and may sometimes **lead only to very slight improvement of a model's performance**. For this reason, it is desired to weigh the impact of tuning on the pipeline of your data analysis project as a whole in order to understand if it is worth pursuing.

## 4. Inspecting RF Hyperparameters in sklearn

To inspect the hyperparameters of a RandomForestRegressor, first, import **RandomForestRegressor** from **sklearn.ensemble** and then instantiate a RandomForestRegressor rf as shown here.

```python
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Set seed for reproducibility
SEED = 1
# Instantiate a random forests regressor 'rf'
rf = RandomForestRegressor(random_state= SEED)
```

## 5. Inspecting RF Hyperparameters in sklearn

The hyperparameters of rf along with their default values can be accessed by calling rf's dot-get_params() method. In the following, we'll be optimizing n_estimators, max_depth, min_samples_leaf and max_features. You can learn more about these hyperparameters by consulting scikit-learn's documentation.

```python
# Inspect rf' s hyperparameters
rf.get_params()
```



## 6. GridSearchCV in sklearn (auto dataset)

We'll perform grid-search cross-validation on the auto-dataset which is already loaded and split into 80%-train and 20%-test. First import mean_squared_error as MSE from sklearn.metrics and GridSearchCV from sklearn.model_selection. Then, define a dictionary called params_rf containing the grid of hyperparameters. Finally, instantiate a GridSearchCV object called grid_rf and pass the parameters rf as estimator, params_rf as param_grid. Also set cv to 3 to perform 3-fold cross-validation. In addition, set scoring to neg_mean_squared_error in order to use negative mean squared error as a metric. Note that the parameter verbose controls verbosity; the higher its value, the more messages are printed during fitting.

```python
# Basic imports
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
# Define a grid of hyperparameter 'params_rf'
params_rf = {
            'n_estimators': [300, 400, 500],
            'max_depth': [4, 6, 8],
            'min_samples_leaf': [0.1, 0.2],
            'max_features': ['log2','sqrt']
}
# Instantiate 'grid_rf'
grid_rf = GridSearchCV(estimator=rf,param_grid=params_rf,cv=3,scoring='neg_mean_squared_error',verbose=1,n_jobs=-1)
```



## 7. Searching for the best hyperparameters

You can now fit grid_rf to the training set as shown here. The output shows messages related to grid fitting as well as the obtained optimal model.

```python
# Fit 'grid_rf' to the training set
grid_rf.fit(X_train, y_train)
```



## 8. Extracting the best hyperparameters

You can extract rf's best hyperparameters by getting the attribute best_params_ from grid_rf. The results are shown here.

```python
# Extract best hyperparameters from 'grid_rf'
best_hyperparams = grid_rf.best_params_
print('Best hyerparameters:\n', best_hyperparams)
```



## 9. Evaluating the best model performance

You can also extract the best model from rf. This enables you to predict the test set labels and evaluate the test-set RMSE. The output shows a result of 3-dot-89. If you would have trained an untuned model, the RMSE would be 3-dot-98.

```python
# Extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_
# Predict the test set labels
y_pred = best_model.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```



## 10. Let's practice!

Now let's try some examples.






















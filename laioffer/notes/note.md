# NoteBook

## what is machine learning?

Machine Learning algorithms use statistics to find patterns in massive amounts of data. And data, here, encompasses a lot of things-numbers, words, clicks, images, what have you. If it is can be digitally stored, it can be fed into a machine learning algorithm.

Machine learning is a subset of artificial intelligence that gives systems the ability to learn and optimize processes without having to be consistently programmed. Simply put, machine learning uses data, statistics, and trial and error to “learn” a specific task without ever having to be specifically coded for the task.

## what is big data?

volumes
velocity
variety

## what is Artificial Intelligence?

Artificial intelligence (AI) is a wide-ranging branch of computer science concerned with building smart machines capable of performing tasks that typically require human intelligence.

## what is Deep Learning?

Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called Artificial Neural Network(ANN)

## Machine Learning vs Deep Learning

Machine Learning has a simpler structure; Deep Learning has a complex structure
Deep Learning requires less human intervention
Deep Learning has hierarchical feature extraction, and suitable for extracting complex features from data.

## Types of Machine Learning

### Supervised Learning

Supervised learning algorithm build a mathematical model of a set of data that contains both the inputs and the desired outputs.

#### Algorithm

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors(KNN)
- Support Vector Machine(SVM)

#### Application

- Fraud detection
- Email spam detection
- Diagnostics
- Image classification
- Risk assessment
- Self-driving

### unsupervised Learning

Unsupervised learning algorithms take a set of data that contains only input, and fin structure in the data, like grouping or clustering data points. The algorithms, therefore, learn from test data that has not been labeled. The algorithm groups the data into various clusters based on their density.

#### Algorithms

- K-means Clustering
- Principal Component Analysis (PCA)
- Latent Dirichlet Allocation (LDA)

#### Application of unsupervised Learning

- Fraud detection
- Email spam detection
- Diagnostics
- Image classification

### Reinforcement Learning

Reinforcement Learning is aiming to reach a goal in a dynamic environment based on several rewards that are provided to it by the system.

#### key factors to reinforcement learning

- Goal
- State
- Actions
- Reward

#### application

- Gaming
- Manufacturing
- Robot navigation

```flow
st=>start: Clearn Data

cond1=>condition: Do you have label data?
cond2=>condition: Types of variables you are predicting
cond3=>condition: Predicting categories

opt1=>operation: Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, etc.
opt2=>operation: Regression: Linear Regression, Ridge Regression, LASSO,etc.
opt3=>operation: Clustering:K-means, Latent Dirichlet Allocation(LDA),etc.
opt4=>operation: Dimensionality Reduction: Principal Component Analysis (PCA),etc

st->cond1
cond1(yes)->cond2
cond2(yes)->opt1
cond2(no)->opt2
cond1(no)->cond3
cond3(yes)->opt3
cond3(no)->opt4


```

## Funfamental of Probability and Statistics

### Joint, Marginal, and Conditional Probability

**Conditional Probability**

$$P(A \mid B)$$
if indenpent:
$$P(A \mid B) = P(A)$$


**Joint Probability** 

$$P(A , B) = P(A \mid B) *P(B) = P(B \mid A)* P(A)$$

if indepent:

$$P(A,B) = P(A) * P(B)$$
$$P(A \mid B) = P(A,B) / P(B)$$

**Chain rule for random variables**

$$\mathrm{P}(X, Y)=\mathrm{P}(X \mid Y) \cdot P(Y)$$

**More than two random variables**
$$\mathrm{P}\left(X_{n}, \ldots, X_{1}\right)=\mathrm{P}\left(X_{n} \mid X_{n-1}, \ldots, X_{1}\right) \cdot \mathrm{P}\left(X_{n-1}, \ldots, X_{1}\right)$$

**example**
$$\begin{aligned}
\mathrm{P}\left(X_{4}, X_{3}, X_{2}, X_{1}\right) &=\mathrm{P}\left(X_{4} \mid X_{3}, X_{2}, X_{1}\right) \cdot \mathrm{P}\left(X_{3}, X_{2}, X_{1}\right) \\
&=\mathrm{P}\left(X_{4} \mid X_{3}, X_{2}, X_{1}\right) \cdot \mathrm{P}\left(X_{3} \mid X_{2}, X_{1}\right) \cdot \mathrm{P}\left(X_{2}, X_{1}\right) \\
&=\mathrm{P}\left(X_{4} \mid X_{3}, X_{2}, X_{1}\right) \cdot \mathrm{P}\left(X_{3} \mid X_{2}, X_{1}\right) \cdot \mathrm{P}\left(X_{2} \mid X_{1}\right) \cdot \mathrm{P}\left(X_{1}\right)
\end{aligned}$$

## Bayes' theorem

In probability theory and statistics, Bayes' theorem describes the probability of event, based on prior knowledge of conditions that might be relatrf to the event.

$$P(A \mid B)=\frac{P(B \mid A) P(A)}{P(B)}$$

> where A and B are events and P(B) $\not =$ 0.
>
> - $P( A \mid B)$ is a conditional probability: the likelihood of event $A$ occurring given that $B$ is true.
>
> - $P(B \mid A)$ is also a conditional probability: the likelihood of event $B$ occurring given that $A$ is true.
>
> - $P(A)$ and $P(B)$ are the proabilities of observing A and respectively; they are known as the marginal probability.
>
> - A and B must be different events.

**Bayes' theorm example**

$$\begin{aligned}
P(\text { User } \mid \text { Positive }) &=\frac{P(\text { Positive } \mid \text { User }) P(\text { User })}{P(\text { Positive })} \\
&=\frac{P(\text { Positive } \mid \text { User }) P(\text { User })}{P(\text { Positive } \mid \text { User }) P(\text { User })+P(\text { Positive } \mid \text { Non-user }) P(\text { Non-user })} \\
&=\frac{0.90 \times 0.05}{0.90 \times 0.05+0.20 \times 0.95}=\frac{0.045}{0.045+0.19} \approx 19 \%
\end{aligned}$$

## Naive Bayes Classifier

### What is Naive Bayes Classifier?

Naive Bayes is a statistical classification technique based on Bayes Theorem. It is one of the simplest supervised learning algorithms.

$$P(h \mid D)=\frac{P(D \mid h) P(h)}{P(D)}$$

- P(h): the probability of hypothesis h being true (regardless of the data). This is known as the **prior probability** of h.
- P(D): the probability of the data (regardless of the hypothesis). This is known as the **prior probability**.
- P(h|D): the probability of hypothesis h given the data D. This is known as **posterior probability**.
- P(D|h): the probability of data d given that the hypothesis h was true. This is known as **posterior probability**.

### How Naive Bayes classifier works?

example:Given an example of weather conditions and playing sports. You need to calculate the probability of playing sports. Now, you need to classify whether players will play or not, based on the weather condition.

#### First Approach (In case of a single feature)

Naive Bayes classifier calculates the probabiliyu of an event in the following steps:

- Step 1: Calculate the prior probability for given class labels
- Step 2: Find Likelihood probability with each attribute for each class
- Step 3: Put these value in Bayes Formula and calculate posterior probability.
- Step 4: See which class has a higher probability, given the input belongs to the higher probability class.

##### Table data

| Whether  | Play |
| -------- | ---- |
| Sunny    | No   |
| Sunny    | No   |
| Overcast | Yes  |
| Rainy    | Yes  |
| Rainy    | Yes  |
| Rainy    | No   |
| Overcast | Yes  |
| Sunny    | No   |
| Sunny    | Yes  |
| Rainy    | Yes  |
| Sunny    | Yes  |
| Overcast | Yes  |
| Overcast | Yes  |
| Rainy    | No   |

##### Frequency Table

| Weather  | No  | Yes |
| -------- | --- | --- |
| Overcast |     | 4   |
| Sunny    | 2   | 3   |
| Rainy    | 3   | 2   |
| Total    | 5   | 9   |

##### Likelihood Table 1

| Weather  |  No   |  Yes  |       |      |
| -------- | :---: | :---: | ----- | ---- |
| Overcast |       |   4   | =4/14 | 0.29 |
| Sunny    |   2   |   3   | =5/14 | 0.36 |
| Rainy    |   3   |   2   | =5/14 | 0.36 |
| Total    |   5   |   9   |       |      |
|          | =5/14 | =9/14 |       |      |
|          | 0.36  | 0.64  |       |      |

##### Likelihood Table2

| Weather  |  No   |  Yes  | Posterior Probability for No | Posterior Probability for Yes |
| -------- | :---: | :---: | :--------------------------: | :---------------------------: |
| Overcast |       |   4   |            0/5=0             |           4/9=0.44            |
| Sunny    |   2   |   3   |           2/5=0.4            |           33/9=0.33           |
| Rainy    |   3   |   2   |           3/5=0.6            |           2/9=0.22            |
| Total    |   5   |   9   |                              |                               |

Now suppose you want to calculate the probability of playing when the weather is overcast.

##### Probability of playing

$$P(Yes \mid Overcast) = \frac{P(Overcast \mid Yes) P(Yes) }{ P (Overcast)}$$

1. Calculate Prior Probabilities:
   $P(Overcast) = 4/14 = 0.29$
   $P(Yes) = 9/14 = 0.64$
2. Calculate Posterior Probabilities:
   $P(Overcast \mid Yes) = 4/9 = 0.44$
3. Put Prior and Posterior probabilities in equation:
   $P (Yes \mid Overcast) = 0.44 * 0.64 / 0.29 = 0.98$(Higher)

##### Probability of not playing

$$P(No \mid Overcast) = \frac{P(Overcast \mid No)  P(No)} {P(Overcast)}$$

1. Calulate Prior Probability:
   $P(Overcast) = 4/14 = 0.29$
   $P(No) = 5/14 = 0.36$
2. Calulate Posterior Probabilities:
   $P(Overcast \mid No) = 0$
3. Put Prior and Posterior probabilities in equation:
   $P(No \mid Overcast) = 0*0.36/0.29 = 0$

#### Second Approach (In case of multiple features)

Naive Bayes classifier calculates the probability of an event which has multiple features in the following steps:

- Step 1: Calculate the prior probability for given class labels
- Step 2: Calculate conditional probability with each attribute for each class
- Step 3: Multiply same class conditional probability
- Step 4: Multiply proor probability with step 3 probability
- Step 5: See which class has higher probability, Higher probabiliry class belongs to given input set step.

| Whether  | Temperature | Play |
| -------- | ----------- | ---- |
| Sunny    | Hot         | No   |
| Sunny    | Hot         | No   |
| Overcast | Hot         | Yes  |
| Rainy    | Mild        | Yes  |
| Rainy    | Cool        | Yes  |
| Rainy    | Cool        | No   |
| Overcast | Cool        | Yes  |
| Sunny    | Mild        | No   |
| Sunny    | Cool        | Yes  |
| Rainy    | Mild        | Yes  |
| Sunny    | Mild        | Yes  |
| Overcast | Mild        | Yes  |
| Overcast | Hot         | Yes  |
| Rainy    | Mild        | No   |

##### Frequency Table (multiple features)

| Weather  | No  | Yes |
| -------- | --- | --- |
| Overcast |     | 4   |
| Sunny    | 2   | 3   |
| Rainy    | 3   | 2   |
| Total    | 5   | 9   |

| Weather | No  | Yes |
| ------- | --- | --- |
| Hot     | 2   | 2   |
| Mild    | 2   | 4   |
| Cold    | 1   | 4   |
| Total   | 5   | 9   |

##### Likelihood Table 1 (multiple features)

| Weather  |  No   |  Yes  |       |      |
| -------- | :---: | :---: | ----- | ---- |
| Overcast |       |   4   | =4/14 | 0.29 |
| Sunny    |   2   |   3   | =5/14 | 0.36 |
| Rainy    |   3   |   2   | =5/14 | 0.36 |
| Total    |   5   |   9   |       |      |
|          | =5/14 | =9/14 |       |      |
|          | 0.36  | 0.64  |       |      |

| Weather |  No   |  Yes  |       |      |
| ------- | :---: | :---: | ----- | ---- |
| Hot     |   2   |   2   | =4/14 | 0.29 |
| Mild    |   2   |   4   | =6/14 | 0.43 |
| Cold    |   1   |   3   | =4/14 | 0.29 |
| Total   |   5   |   9   |       |      |
|         | =5/14 | =9/14 |       |      |
|         | 0.36  | 0.64  |       |      |


##### Likelihood Table2 (multiple features)

| Weather  |  No   |  Yes  | Posterior Probability for No | Posterior Probability for Yes |
| -------- | :---: | :---: | :--------------------------: | :---------------------------: |
| Overcast |       |   4   |            0/5=0             |           4/9=0.44            |
| Sunny    |   2   |   3   |           2/5=0.4            |           3/9=0.33            |
| Rainy    |   3   |   2   |           3/5=0.6            |           2/9=0.22            |
| Total    |   5   |   9   |                              |                               |

| Weather |  No   |  Yes  | Posterior Probability for No | Posterior Probability for Yes |
| ------- | :---: | :---: | :--------------------------: | :---------------------------: |
| Hot     |   2   |   2   |           2/5=0.4            |           2/9=0.22            |
| Mild    |   2   |   4   |           2/5=0.4            |           4/9=0.44            |
| Cold    |   1   |   3   |           1/5=0.2            |           3/9=0.33            |
| Total   |   5   |   9   |                              |                               |

Now suppose you want to calculate the probability of playing when the weather is overcast, and the temperature is mild.

##### Probability of playing

$$P(Play= Yes | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play= Yes)P(Play=Yes) ..........(1)$$
$$PP(Weather=Overcast, Temp=Mild | Play= Yes)= P(Overcast |Yes) P(Mild |Yes) ………..(2)$$

1. Calculate Prior Probailities:
   $P(Yes) = 9/14 = 0.64$
2. Calculate Posterior Probabilities:
   $P(Overcast \mid Yes) = 4/9 = 0.44 \\P(Mild \mid Yes) = 4/9 = 0.44$
3. Put Posterior probabilities in equation (2):
   $P(Weather=Overcast, Temp=Mild \mid Play= Yes) = P(Weather=Overcast \mid Play= Yes)P(Temp=Mild \mid Play= Yes) = 0.44 * 0.44 = 0.1936(Higher)$
4. Put Prior and Posterior probabilities in equation (1):
   $P(Play= Yes \mid Weather=Overcast, Temp=Mild) = 0.1936*0.64 = 0.124$

##### Probability of not playing

$$P(Play= No | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play= No)P(Play=No) ..........(1)$$
$$PP(Weather=Overcast, Temp=Mild | Play= No)= P(Overcast |No) P(Mild |No) ………..(2)$$

1. Calculate Prior Probailities:
   $P(No) = 9/14 = 0.36$
2. Calculate Posterior Probabilities:
   $P(Overcast \mid No) = 0/5 = 0 \\
   P(Mild \mid No) = 2/5 = 0.4$
3. Put Posterior probabilities in equation (2):
   $P(Weather=Overcast, Temp=Mild \mid Play= No) \\
   = P(Weather=Overcast \mid Play= No)P(Temp=Mild \mid Play= No) \\
   = 0 * 0.4 = 0(Higher)$
4. Put Prior and Posterior probabilities in equation (1):
   $P(Play= No \mid Weather=Overcast, Temp=Mild) = 0*0.36 = 0$

The probability of a 'Yes' class is higher. So you can say here that if the weather is overcast than players will play the sport.


### Advantages

- It is not only a simple approach but also a fast and accurate method for prediction.
- Naive Bayes has very low computation cost.
- It can efficiently work on a large dataset.
- It performs well in case of discrete response variable compared to the continuous variable.
- It can be used with multiple class prediction problems.
- It also performs well in the case of text analytics problems.
- When the assumption of independence holds, a Naive Bayes classifier performs better compared to other models like logistic regression.

### Disadvantages

- The assumption of independent features. In practice, it is almost impossible that model will get a set of predictors which are entirely independent.
- If there is no training tuple of a particular class, this causes zero posterior probability. In this case, the model is unable to make predictions. This problem is known as Zero Probability/Frequency Problem.

## Random Variable and Probability Distribution

### randomness

### Random variable
In probability and statistics, a random variable, random quantity, aleatory variable, or stochastic variable is described informally as a variable whose values depend on outcomes of a random phenomenon

#### Parameter

#### types of random variables

1. Discrete random variable
2. Continuous random variable

### discrete random variable

A discrete random variable is one which may take on only a countable number of distinct values

#### Expected value (of a discrete random variable)

The population mean for a random variable and is therefore a measure of centre for the distribution of a random variable.

The expected value of random variable X is often written as E(X) or µ or µX.

The expected value is the **‘long-run mean’** in the sense that, if as more and more values of the random variable were collected (by sampling or by repeated trials of a probability activity), the sample mean becomes **closer to the expected value**.

For a discrete random variable the expected value is calculated by summing the product of the value of the random variable and its associated probability, taken over all of the values of the random variable.

In symbols,
$$\mu=E(X)=\sum_{i=1}^{n} x_{i} p_{i}$$

#### Variance (of a discrete random variable)

**A measure of spread for a distribution of a random variable** that determines the degree to which the values of a random variable differ from the expected value.

The variance of random variable X is often written as Var(X) or σ2 or σ2x.

For a discrete random variable the variance is calculated by summing the product of the square of the difference between the value of the random variable and the expected value, and the associated probability of the value of the random variable, taken over all of the values of the random variable.

For a discrete random variable X, the variance of X is obtained as follows:
 $$\sigma^{2}=V \operatorname{ar}(X)=\sum_{i=1}^{n}\left(x_{i}-\mu\right)^{2} p_{i}$$

 where the sum is taken over all values of x for which pX(x)>0. So the variance of X is the weighted average of the squared deviations from the mean μ, where the weights are given by the probability function pX(x) of  X

An equivalent formula is, $\operatorname{Var}(X)=\mathrm{E}\left[(X)^{2}\right] - \mathrm{E}\left[(X)\right]^{2}$

The variance of a random variable $X$ is the expected value of the squared deviation from the mean of $X$, $\mu=\mathrm{E}\left[X\right]$:
$$\operatorname{Var}(X)=\mathrm{E}\left[(X-\mu)^{2}\right]$$

The square root of the variance is equal to the standard deviation.

### Discrete Probability Distribution

In probability theory and statistics, a probability distribution is the mathematical function that gives the probabilities of occurrence of different possible outcomes for an experiment.

#### Bernoulli distribution

##### Properties

If $X$ is a random variable with this distribution, then
$$\operatorname{Pr}(X=1)=p=1-\operatorname{Pr}(X=0)=1-q$$

The probability mass function {\displaystyle f}f of this distribution, over possible outcomes k, is

$$f(k ; p)=\left\{\begin{array}{ll}
p & \text { if } k=1 \\
q=1-p & \text { if } k=0
\end{array}\right.$$
This can also be expressed as
$$f(k ; p)=p^{k}(1-p)^{1-k} \quad \text { for } k \in\{0,1\}$$

or as
$$f(k ; p)=p k+(1-p)(1-k) \quad \text { for } k \in\{0,1\}$$

##### Mean

The expected value of a Bernoulli random variable $X$ is

$$\mathrm{E}(X)=p$$

This is due to the fact that for a Bernoulli distributed random variable  X  with  $\operatorname{Pr}(X=1)=p$  and  $\operatorname{Pr}(X=0)=q$  we find

$$\mathrm{E}[X]=\operatorname{Pr}(X=1) \cdot 1+\operatorname{Pr}(X=0) \cdot 0=p \cdot 1+q \cdot 0=p .^{[2]}$$

##### Variance

The variance of a Bernoulli distributed  $X$  is

$$\operatorname{Var}[X]=p q=p(1-p)$$

We first find
$$\mathrm{E}\left[X^{2}\right]=\operatorname{Pr}(X=1) \cdot 1^{2}+\operatorname{Pr}(X=0) \cdot 0^{2}=p \cdot 1^{2}+q \cdot 0^{2}=p$$

From this follows

$$\operatorname{Var}[X]=\mathrm{E}\left[X^{2}\right]-\mathrm{E}[X]^{2}=p-p^{2}=p(1-p)=p q^{[2]}$$

#### Binomial distribution

$$P(X=x)=C_{n}^{x} p^{x} q^{n-x}, x=0,1,2, \ldots, n$$

### Continuous random variable

Formally, a continuous random variable is a random variable whose cumulative distribution function is continuous everywhere.

For any continuous random variable with probability density function f(x), we have that:

$$\int_{a}^{b} f(x) d x=P(a \leq X \leq b)$$

If X is a continuous random variable with p.d.f. f(x) defined on a ≤ x ≤ b, then the cumulative distribution function (c.d.f.), written F(t) is given by:
$$\mathrm{F}(\mathrm{t})=\mathrm{P}(\mathrm{X} \leq \mathrm{t})=\int_{\mathrm{a}}^{\mathrm{t}} \mathrm{f}(\mathrm{x}) \mathrm{d} \mathrm{x}$$

Expected:$$\mu=E(X)=\int_{-\infty}^{+\infty} x f(x) d x$$

Variance of continue random variable:
$$\sigma^{2}=V \operatorname{ar}(X)=\int_{-\infty}^{+\infty}[x-E(X)]^{2} f(x) d x$$

### Continuous Probability Distribution

#### Normal Distribution

In probability theory, a normal (or Gaussian or Gauss or Laplace–Gauss) distribution is a type of continuous probability distribution for a real-valued random variable.

Formula
$$f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}}$$

$f(x)$ = probability density function
$\sigma$ = standard deviation
$\mu$ = mean


## Central Limit Theorem

In probability theory, the central limit theorem (CLT) establishes that, in many situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed.

## What is Linear Regression?

In statistic, linear rehression is a linear approach to modelling hre ewlationship between a scalar response and one or more explanatoey variables(also known as dependent and independent cariables).

### Simple and multiple linear regression

The basic model for multiple linear regression is

$$Y_{i}=\beta_{0}+\beta_{1} X_{i 1}+\beta_{2} X_{i 2}+\ldots+\beta_{p} X_{i p}+\epsilon_{i}$$

for each observation i = 1, ... , n.

## What is Loss Function?

Machine learn by means of al loss funcrion. It's a method of evaluation how well specific alhorithm models the given data. If oreductuibs devuates too muvh fromactual results, loss function would cough up a very large number.

There’s no one-size-fits-all loss function to algorithms in machine learning.

loss functions can be classified into two major categories depending upon the type of learning task we are dealing with — **Regression losses** and **Classification losses**.

### Regression Losses

Mean Square Error/Quadratic Loss/L2 Loss

Mathematical formulation :Mean Squared Error 均方差

$$M S E=\frac{\sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}}{n}$$

As the name suggests, Mean square error is measured as the average of squared difference between predictions and actual observations. It’s only **concerned with the average magnitude(大小) of error** **irrespective** of their **direction**.


## What is Cost Function?

It is a function that *measures the performance of a Machine Learning model* for given data. Cost Function quantifies the error between predicted values and expected values and *presents it in the form of a single real number*. 

The purpose of Cost Function is to be either:

- Minimized - then returned value is usually called cost, loss or error. The goal is to find the values of model parameters for which Cost Function return as small number as possible.
  
- Maximized - then the value it yields is named a reward. The goal is to find values of model parameters for which returned number is as large as possible.

## What is the difference between a cost function and a loss function in machine learning?

The terms cost and loss functions almost refer to the same meaning. But, loss function mainly applies for a single training set as compared to the cost function which deals with a penalty for a number of training sets or the complete batch. It is also sometimes called an error function.

In short, we can say that the loss function is a part of the cost function. The cost function is calculated as **an average of loss functions**. The loss function is a value which is **calculated at every instance**. 

So, for a single training cycle loss is calculated numerous times, but the cost function is only calculated once.


## Loss Function for Linear Regression

SSE

## Least Squares

The method of least squares is a standard approach in regression analysis to approximate the solution of overdetermined systems (sets of equations in which there are more equations than unknowns) by minimizing the sum of the squares of the residuals made in the results of every single equation.

The most important application is in data fitting


### How to calculate linear regression using least square method

Let $X$ be the independent variable and $Y$ be the dependent variable. We will define a linear relationship between these two variables as follows:
$$Y=m X+c$$

Our challenege today is to determine the value of m and c, that gives the minimum error for the given dataset. We will be doing this by using the Least Squares method.

A **loss function** in machine learning is simply a measure of how different the predicted value is from the actual value.

Today we will be using the **Quadratic Loss Function** to calculate the loss or error in our model. It can be defined as:
 $$L(x)=\sum_{i=1}^{n}\left(y_{i}-p_{i}\right)^{2}$$

![pic](https://raw.githubusercontent.com/wanghaonanlpc/Figure-bed/master/20201215112825.jpeg)

Now that we have determined the loss function, the only thing left to do is minimize it. This is done by finding the **partial derivative** of **L**, equating it to 0 and then finding an expression for **m** and **c**. After we do the math, we are left with these equations:
 $$\begin{array}{c}
m=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \\\\
c=\bar{y}-m \bar{x}
\end{array}$$

$$\mathbf{m}=\frac{N \Sigma(x y)-\Sigma x \Sigma y}{N \Sigma\left(x^{2}\right)-(\Sigma x)^{2}}$$

## Maximum Likelihood Estimation(最大似然估计)

In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable.

Maximum likelihood estimation is a method that determines values for the parameters of a model. The parameter values are found such that they maximise the likelihood that the process described by the model produced the data that were actually observed.

### Calculating the Maximum Likelihood Estimates

The probability density of observing a single data point x, that is generated from a Gaussian distribution is given by:

$$P(x ; \mu, \sigma)=\frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right)$$

Suppose we have three data points this time and we assume that they have been generated from a process that is adequately described by a Gaussian distribution. These **points** are **9**, **9.5** and **11**.

The semi colon used in the notation P(x; μ, σ) is there to emphasise that the symbols that appear after it are parameters of the probability distribution. So it shouldn’t be confused with a **conditional probability** (which is typically represented with a vertical line e.g. **P(A| B)**).

In our example the total (joint) probability density of observing the three data points is given by:

$$\begin{array}{r}
P(9,9.5,11 ; \mu, \sigma)=\frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{(9-\mu)^{2}}{2 \sigma^{2}}\right) \times \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{(9.5-\mu)^{2}}{2 \sigma^{2}}\right) \times \frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{(11-\mu)^{2}}{2 \sigma^{2}}\right)
\end{array}$$

We just have to figure out the values of μ and σ that results in giving the maximum value of the above expression.

If you’ve covered calculus in your maths classes then you’ll probably be aware that there is a technique that can help us find maxima (and minima) of functions. It’s called **differentiation**. I’ll assume that the reader knows **how to perform differentiation** on common functions.

### The log likelihood

The above expression for the total probability is actually quite a pain to differentiate, so it is almost always simplified by taking the natural logarithm of the expression. This is absolutely fine because the natural logarithm is a monotonically increasing function. This means that if the value on the x-axis increases, the value on the y-axis also increases

ln algorithms:

$$\begin{aligned}
&\ln (M N)=\ln M+\ln N\\
&\ln (M / N)=\ln M-\ln N\\
&\ln \left(M^{\wedge} n\right)=n \ln M\\
&\ln 1=0\\
&\ln e =1
\end{aligned}$$

Taking logs of the original expression gives us:

$$\begin{aligned}
\ln (P(x ; \mu, \sigma))=\ln \left(\frac{1}{\sigma \sqrt{2 \pi}}\right)-\frac{(9-\mu)^{2}}{2 \sigma^{2}}+\ln \left(\frac{1}{\sigma \sqrt{2 \pi}}\right)-& \frac{(9.5-\mu)^{2}}{2 \sigma^{2}}
&+\ln \left(\frac{1}{\sigma \sqrt{2 \pi}}\right)-\frac{(11-\mu)^{2}}{2 \sigma^{2}}
\end{aligned}$$

This expression can be simplified again using the laws of logarithms to obtain:

$$\ln (P(x ; \mu, \sigma))=-3 \ln (\sigma)-\frac{3}{2} \ln (2 \pi)-\frac{1}{2 \sigma^{2}}\left[(9-\mu)^{2}+(9.5-\mu)^{2}+(11-\mu)^{2}\right]$$

**This expression can be differentiated to find the maximum.** In this example we’ll find the MLE of the mean, $\mu$. To do this we take the partial **derivative** of the function with respect to $\mu$ , giving

$$\frac{\partial \ln (P(x ; \mu, \sigma))}{\partial \mu}=\frac{1}{\sigma^{2}}[9+9.5+11-3 \mu]$$

Finally, setting the left hand side of the equation to zero and then rearranging for μ gives:

$$\mu=\frac{9+9.5+11}{3}=9.833$$

And there we have our maximum likelihood estimate for $\mu$. We can do the same thing with $\sigma$ too.

[原文地址：https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)


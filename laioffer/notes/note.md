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
| Rainy    |   3   |   2   |           3/5=0.6            |           3/9=0.22            |
| Total    |   5   |   9   |                              |                               |

Now suppose you want to calculate the probability of playing when the weather is overcast.

Probability of playing:
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





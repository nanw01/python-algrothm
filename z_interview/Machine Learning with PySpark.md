# Machine Learning with PySpark

## 1. Characteristics of Spark.

Spark is currently the most popular technology for processing large quantities of data. Not only is it able to handle enormous data volumes, but it does so very efficiently too! Also, unlike some other distributed computing technologies, developing with Spark is a pleasure.

Which of these describe Spark?

Spark is a framework for cluster computing.

- Spark is a framework for cluster computing.

- Spark does most processing in memory.

- Spark has a high-level API, which **conceals** a lot of complexity.

  

### Components in a Spark Cluster

Spark is a distributed computing platform. It achieves efficiency by distributing data and computation across a cluster of computers.

A Spark cluster consists of a number of hardware and software **components** which work together.



Which of these is part of a Spark cluster?

- One or more nodes
- A cluster manager
- Executors

![](https://tva1.sinaimg.cn/large/008eGmZEly1gohtp0e6vkj30ms0isq4o.jpg)



### Connecting to Spark



Location of Spark maste

```python

import pyspark

pyspark.__version__

```



Sub-Modules

In addition to pyspark there are

Structured Data — **pyspark.sql**

Streaming Data — **pyspark.streaming**

Machine Learning — **pyspark.mllib (deprecated) and pyspark.ml**


### Spark URL

Remote Cluster using Spark URL — spark://<IP address | DNS name>:<port>

Example:

spark://13.59.151.161:7077

Local Cluster

Examples:

local — only 1 core;

local[4] — 4 cores; or

local[*] — all available cores.



Which of the following is a valid way to specify the location of a Spark cluster?



- `spark://13.59.151.161:7077`
- `spark://ec2-18-188-22-23.us-east-2.compute.amazonaws.com:7077`
- `local`
- `local[4]`
- `local[*]`



```python

# Import the PySpark module
from pyspark.sql import SparkSession

# Create SparkSession object
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('test') \
                    .getOrCreate()

# What version of Spark?
# (Might be different to what you saw in the presentation!)
print(spark.version)

# Terminate the cluster
spark.stop()

```





### Loading Data



Notes on CSV format:

- fields are separated by a comma (this is the default separator) and
- missing data are denoted by the string 'NA'.

Data dictionary:

- `mon` — month (integer between 1 and 12)
- `dom` — day of month (integer between 1 and 31)
- `dow` — day of week (integer; 1 = Monday and 7 = Sunday)
- `org` — origin airport ([IATA code](https://en.wikipedia.org/wiki/IATA_airport_code))
- `mile` — distance (miles)
- `carrier` — carrier ([IATA code](https://en.wikipedia.org/wiki/List_of_airline_codes))
- `depart` — departure time (decimal hour)
- `duration` — expected duration (minutes)
- `delay` — delay (minutes)





```python

# Read data from CSV file
flights = spark.read.csv('flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
print(flights.dtypes)

```

Selected methods:

**count()**

**show()**

**printSchema()**

Selected attributes:

**dtypes**



### Loading SMS spam data



[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).



Notes on CSV format:

- no header record and
- fields are separated by a semicolon (this is **not** the default separator).

Data dictionary:

- `id` — record identifier
- `text` — content of SMS message
- `label` — spam or ham (integer; 0 = ham and 1 = spam)



```python

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv('sms.csv', sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()

```



### Specify column types

```python

schema = StructType([
StructField("maker", StringType()),
StructField("model", StringType()),
StructField("origin", StringType()),
StructField("type", StringType()),
StructField("cyl", IntegerType()),
StructField("size", DoubleType()),
StructField("weight", IntegerType()),
StructField("length", DoubleType()),
StructField("rpm", IntegerType()),
StructField("consumption", DoubleType())
])
cars = spark.read.csv("cars.csv", header=True, schema=schema, nullValue='NA')

```



## 2. Classification

### Removing columns and rows

In this exercise you need to trim those data down by:

1. removing an uninformative column and
2. removing rows which do not have information about whether or not a flight was delayed.

The data are available as `flights`.



```python

# Remove the 'flight' column
flights_drop_column = flights.drop('flight')

# Number of records with missing 'delay' values
flights_drop_column.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights_valid_delay = flights_drop_column.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights_none_missing = flights_valid_delay.dropna()
print(flights_none_missing.count())

```



### Column manipulation



The next step of preparing the flight data has two parts:

1. convert the units of distance, replacing the `mile` column with a `km`column; and
2. create a Boolean column indicating whether or not a flight was delayed.





```python

# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)

```



### Categorical columns



In the flights data there are two columns, `carrier` and `org`, which hold categorical data. You need to transform those columns into indexed numerical values.



```python

from pyspark.ml.feature import StringIndexer

# Create an indexer
indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx').fit(flights_indexed).transform(flights_indexed)

```



### Assembling columns



An *updated* version of the `flights` data, which takes into account all of the changes from the previous few exercises, has the following predictor columns:

- `mon`, `dom` and `dow`
- `carrier_idx` (indexed value from `carrier`)
- `org_idx` (indexed value from `org`)
- `km`
- `depart`
- `duration`

```python

# Import the necessary class
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols=[
    'mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart', 'duration'
], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)

```



### Decision Tree



#### 1. Decision Tree

Your first Machine Learning model will be a Decision Tree. This is probably the most intuitive model, so it seems like a good place to start.

#### 2. Anatomy of a Decision Tree: Root node

A Decision Tree is constructed using an algorithm called **"Recursive Partitioning"**. Consider a hypothetical example in which you build a Decision Tree to divide data into two classes, green and blue. You start by putting all of the records into the root node. Suppose that there are more green records than blue, in which case this node will be labelled "green". Now from amongst the predictors in the data you need to choose the one that will result in the most informative split of the data into two groups. Ideally you want the groups to be as homogeneous (or "pure") as possible: one should be mostly green and the other should be mostly blue.

#### 3. Anatomy of a Decision Tree: First split

Once you have identified the most informative predictor, you split the data into two sets, labeled "green" or "blue" according to the dominant class. And this is where the recursion kicks in: you then apply exactly the same procedure on each of the child nodes, selecting the most informative predictor and splitting again.

#### 4. Anatomy of a Decision Tree: Second split

So, for example, the green node on the left could be split again into two groups.

#### 5. Anatomy of a Decision Tree: Third split

And the resulting green node could once again be split. The depth of each branch of the tree need not be the same. There are a variety of stopping criteria which can cause splitting to stop along a branch. For example, if the number of records in a node falls below a threshold or the purity of a node is above a threshold, then you might stop splitting. Once you have built the Decision Tree you can use it to make predictions for new data by following the splits from the root node along to the tip of a branch. The label for the final node would then be the prediction for the new data.

#### 6. Classifying cars

Let's make this more concrete by looking at the cars data. You've transformed the country of origin column into a numeric index called 'label', with zero corresponding to cars manufactured in the USA and one for everything else. The remaining columns have all been consolidated into a column called 'features'. You want to build a Decision Tree which will use "features" to predict "label".

#### 7. Split train/test

An important aspect of building a Machine Learning model is being able to assess how well it works. In order to do this we use the randomSplit() method to randomly split our data into two sets, a training set and a testing set. The proportions may vary, but generally you're looking at something like an 80:20 split, which means that the training set ends up having around 4 times as many records as the testing set.

#### 8. Build a Decision Tree model

Finally the moment has come, you're going to build a Decision Tree. You start by creating a **DecisionTreeClassifier()** object. The next step is to fit the model to the training data by calling the **fit()** method.

#### 9. Evaluating

Now that you've trained the model you can assess how effective it is by making predictions on the test set and comparing the predictions to the known values. The transform() method adds new columns to the DataFrame. The prediction column gives the class assigned by the model. You can compare this directly to the known labels in the testing data. Although the model gets the first example wrong, it's correct for the following four examples. There's also a probability column which gives the probabilities assigned to each of the outcome classes. For the first example, the model predicts that the outcome is 0 with probability 96%.

#### 10. Confusion matrix

A good way to understand the performance of a model is to create a confusion matrix which gives a breakdown of the model predictions versus the known labels. The confusion matrix consists of four counts which are labelled as follows: - "positive" indicates a prediction of 1, while - "negative" indicates a prediction of 0 and - "true" corresponds to a correct prediction, while - "false" designates an incorrect prediction. In this case the true positives and true negatives dominate but the model still makes a number of incorrect predictions. These counts can be used to calculate the accuracy, which is the proportion of correct predictions. For our model the accuracy is 74%.

#### 11. Let's build Decision Trees!

So, now that you know how to build a Decision Tree model with Spark, you can try that out on the flight data.





```python

# Import the Decision Tree Classifier class

from pyspark.ml.classification import DecisionTreeClassifier

```





### Train/test split



You will split the data into two components:

- training data (used to train the model) and
- testing data (used to test the model).



```python

# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([.8,.2], seed=17)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights.count()
print(training_ratio)

```



### Build a Decision Tree



The data are available as `flights_train` and `flights_test`.



```python

# Import the Decision Tree Classifier class
from pyspark.ml.classification import DecisionTreeClassifier

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)

```





### Evaluate the Decision Tree



A ***confusion matrix*** gives a useful breakdown of predictions versus known values. It has four cells which represent the counts of:

- *True Negatives* (TN) — model predicts negative outcome & known outcome is negative
- *True Positives* (TP) — model predicts positive outcome & known outcome is positive
- *False Negatives* (FN) — model predicts negative outcome but known outcome is positive
- *False Positives* (FP) — model predicts positive outcome but known outcome is negative.



```python

# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label != prediction').count()
FP = prediction.filter('prediction = 1 AND label != prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FN + FP)
print(accuracy)

```



**Accuracy**, **Recall**,**Precision**,**F-measure**



### Logistic Regression



#### 1. Logistic Regression

You've learned to build a Decision Tree. But it's good to have options. Logistic Regression is another commonly used **classification model**.

#### 2. Logistic Curve

It uses a logistic function to model a binary target, where the target states are usually denoted by 1 and 0 or TRUE and FALSE. The maths of the model are outside the scope of this course, but this is what the logistic function looks like. **For a Logistic Regression model the x-axis is a linear combination of predictor variables and the y-axis is the output of the model**. Since the value of the logistic function is a number between zero and one, it's often thought of as a probability. In order to translate this number into one or other of the target states it's compared to a threshold, which is normally set at one half.

#### 3. Logistic Curve

**If the number is above the threshold then the predicted state is one.**

#### 4. Logistic Curve

**Conversely, if it's below the threshold then the predicted state is zero**. The model derives coefficients for each of the numerical predictors. Those coefficients might...

#### 5. Logistic Curve

shift the curve to the right...

#### 6. Logistic Curve

or to the left. They might make the transition between states...

#### 7. Logistic Curve

more gradual...

#### 8. Logistic Curve

or more rapid. These characteristics are all extracted from the training data and will vary from one set of data to another.

#### 9. Cars revisited

Let's make this more concrete by returning to the cars data. You'll focus on the numerical predictors for the moment and return to categorical predictors later on. As before you prepare the data by consolidating the predictors into a single column and then randomly splitting the data into training and testing sets.

#### 10. Build a Logistic Regression model

To build a Logistic Regression model you first need to import the associated class and then create a classifier object. This is then fit to the training data using the fit() method.

#### 11. Predictions

With a trained model you are able to make predictions on the testing data. As you saw with the Decision Tree, the transform() method adds the prediction and probability columns. The probability column gives the predicted probability of each class, while the prediction column reflects the predicted label, which is derived from the probabilities by applying the threshold mentioned earlier.

#### 12. Precision and recall

You can assess the quality of the predictions by forming a confusion matrix. The quantities in the cells of the matrix can then be used to form some informative ratios. Recall that a positive prediction indicates that a car is manufactured outside of the USA and that predictions are considered to be true or false depending on whether they are correct or not. Precision is the proportion of positive predictions which are correct. For your model, two thirds of predictions for cars manufactured outside of the USA are correct. Recall is the proportion of positive targets which are correctly predicted. Your model also identifies 80% of cars which are actually manufactured outside of the USA. Bear in mind that these metrics are based on a relatively small testing set.

#### 13. Weighted metrics

Another way of looking at these ratios is to weight them across the positive and negative predictions. You can do this by creating an evaluator object and then calling the **evaluate()** method. This method accepts an argument which specifies the required metric. It's possible to request the weighted precision and recall as well as the overall accuracy. It's also possible to get the **F1 metric,** the harmonic mean of precision and recall, which is generally more robust than the accuracy. All of these metrics have assumed a threshold of one half. What happens if you vary that threshold?

#### 14. ROC and AUC

**A threshold is used to decide whether the number returned by the Logistic Regression model translates into either the positive or the negative class.** **By default that threshold is set at a half**. However, this is not the only choice. Choosing a larger or smaller value for the threshold will affect the performance of the model. **The ROC curve plots the true positive rate versus the false positive rate as the threshold increases from zero (top right) to one (bottom left).** **The AUC summarizes the ROC curve in a single number.** **It's literally the area under the ROC curve.** AUC indicates how well a model performs across all values of the threshold. An ideal model, that performs perfectly regardless of the threshold, would have AUC of 1. In an exercise we'll see how to use another evaluator to calculate the AUC.

#### 15. Let's do Logistic Regression!

You now know how to build a Logistic Regression model and assess the performance of that model using various metrics. Let's give this a try

```python
from pyspark.ml.classification import LogisticRegression
```

### Build a Logistic Regression model

The data have been split into training and testing sets and are available as `flights_train` and `flights_test`

```python

# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy('label', 'prediction').count().show()

```



### Evaluate the Logistic Regression model



Accuracy is generally not a very reliable metric because it can be biased by the most common target class.

There are two other useful metrics:

- *precision* and
- *recall*.



Precision is the proportion of positive predictions which are correct. 

Recall is the proportion of positives outcomes which are correctly predicted. 



The precision and recall are generally formulated in terms of the positive target class. But it's also possible to calculate *weighted* versions of these metrics which look at both target classes.

The components of the confusion matrix are available as `TN`, `TP`, `FN` and `FP`, as well as the object `prediction`.



```python

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: "areaUnderROC"})


```



### Turning Text into Tables



#### 1. Turning Text into Tables

It's said that 80% of Machine Learning is data preparation. As we'll see in this lesson, this is particularly true for text data. Before you can use Machine Learning algorithms you need to take unstructured text data and create structure, ultimately transforming the data into a table.

#### 2. One record per document

We start with a collection of documents. These documents might be anything from a short snippet of text, like an SMS or email, to a lengthy report or book. Each document will become a record in the table.

#### 3. One document, many columns

The text in each document will be mapped to columns in the table. First the text is split into words or tokens. You then remove short or common words that do not convey too much information. The table will then indicate the number of times that each of the remaining words occurred in the text. This table is also known as a "term-document matrix". There are some nuances to the process, but that's the central idea.

#### 4. A selection of children's books

Suppose that your documents are the names of children's books. The raw data might look like this. Your job will be to transform these data into a table with one row per document and a column for each of the words.

#### 5. Removing punctuation

You're interested in words, not punctuation. You'll use regular expressions (or REGEX), a mini-language for pattern matching, to remove the punctuation symbols. Regular expressions is another big topic and outside of the scope of this course, but basically you are giving a list of symbols or text pattern to match. The hyphen is escaped by the backslashes because it has another meaning in the context of regular expressions. By escaping it you tell Spark to interpret the hyphen literally. You need to specify a column name, books.text, a pattern to be matched (stored in the variable REGEX), and the replacement text, which is simply a space. You now have some double spaces but you can use REGEX to clean those up too.

#### 6. Text to tokens

**Next you split the text into words or tokens.** You create a tokenizer object, giving it the name of the input column containing the text and the output column which will contain the tokens. The tokenizer is then applied to the text using the transform() method. In the results you see a new column in which each document has been transformed into a list of words. As a side effect the words have all been reduced to lower case.

#### 7. What are stop words?

**Some words occur frequently in all of the documents.** These common or "stop" words convey very little information, so you will also remove them using an instance of the StopWordsRemover class. This contains a list of stop words which can be customized if necessary.

#### 8. Removing stop words

Since you didn't give the input and output column names earlier, you specify them now and then apply the transform method. You could also have given these names when you created the remover.

#### 9. Feature hashing

Your documents might contain a large variety of words, so in principle our table could end up with an enormous number of columns, many of which would be only sparsely populated. It would also be handy to convert the words into numbers. Enter the hashing trick, which in simple terms converts words into numbers. You create an instance of the HashingTF class, providing the names of the input and output columns. You also give the number of features, which is effectively the largest number that will be produced by the hashing trick. This needs to be sufficiently big to capture the diversity in the words. The output in the hash column is presented in sparse format, which we will talk about more later on. For the moment though it's enough to note that there are two lists. The first list contains the hashed values and the second list indicates how many times each of those values occurs. For example, in the first document the word "long" has a hash of 8 and occurs twice. Similarly, the word "five" has a hash of 6 and occurs once in each of the last two documents.

#### 10. Dealing with common words

The final step is to account for some words occurring frequently across many documents. If a word appears in many documents then it's probably going to be less useful for building a classifier. We want to weight the number of counts for a word in a particular document against how frequently that word occurs across all documents. To do this you reduce the effective count for more common words, giving what is known as the "inverse document frequency". Inverse document frequency is generated by the IDF class, which is first fit to the hashed data and then used to generate weighted counts. The word "five", for example, occurs in multiple documents, so its effective frequency is reduced. Conversely, the word "long" only occurs in one document, so its effective frequency is increased.

#### 11. Text ready for Machine Learning!

The inverse document frequencies are precisely what we need for building a Machine Learning model. Let's do that with the SMS data.



### Punctuation, numbers and tokens



But first you'll need to prepare the SMS messages as follows:

- remove punctuation and numbers
- tokenize (split into individual words)
- remove stop words
- apply the hashing trick
- convert to TF-IDF representation.

In this exercise you'll remove punctuation and numbers, then tokenize the messages.

The SMS data are available as `sms`.



```python

# Import the necessary functions
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, ' +', ' '))

# Split the text into words
wrangled = Tokenizer(inputCol='text', outputCol='words').transform(wrangled)

wrangled.show(4, truncate=False)

```



### Stop words and hashing



A quick reminder about these concepts:

- The hashing trick provides a fast and space-efficient way to map a very large (possibly infinite) set of items (in this case, all words contained in the SMS messages) onto a smaller, finite number of values.
- The TF-IDF matrix reflects how important a word is to each document. It takes into account both the frequency of the word within each document but also the frequency of the word across all of the documents in the collection.

The tokenized SMS data are stored in `sms` in a column named `words`.



```python

from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF

# Remove stop words.
wrangled = StopWordsRemover(inputCol='words', outputCol='terms')\
      .transform(sms)

# Apply the hashing trick
wrangled = HashingTF(inputCol='terms', outputCol='hash', numFeatures=1024)\
      .transform(wrangled)

# Convert hashed symbols to TF-IDF
tf_idf = IDF(inputCol='hash',outputCol='features')\
      .fit(wrangled).transform(wrangled)
      
tf_idf.select('terms', 'features').show(4, truncate=False)

```







### Training a spam classifier



The SMS data have now been prepared for building a classifier. Specifically, this is what you have done:

- removed numbers and punctuation
- split the messages into words (or "tokens")
- removed stop words
- applied the hashing trick and
- converted to a TF-IDF representation.



Next you'll need to split the TF-IDF data into training and testing sets.



```python

# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([.8,.2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2).fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy('label', 'prediction').count().show()

```



## 3. Regression

### One-Hot Encoding

#### 1. One-Hot Encoding

In the last chapter you saw how to use categorical variables in a model by simply converting them to indexed numerical values. In general this is not sufficient for a regression model. Let's see why.

#### 2. The problem with indexed values

In the cars data the type column is categorical, with six levels: 'Midsize', 'Small', 'Compact', 'Sporty', 'Large' and 'Van'. Here you can see the number of times that each of those levels occurrs in the data. You used a string indexer to assign a numerical index to each level. However, there's a problem with the index: **the numbers don't have any objective meaning.** The index for 'Sporty' is 3. Does it make sense to do arithmetic on that index? No. For example, it wouldn't be meaningful to add the index for 'Sporty' to the index for 'Compact'. Nor would it be valid to compare those indexes and say that 'Sporty' is larger or smaller than 'Compact'. However, a regression model works by doing precisely this: **arithmetic on predictor variables**. You need to convert the index values into a format in which you can **perform meaningful mathematical operations.**

#### 3. Dummy variables

The first step is to create a column for each of the levels. Effectively you then place a check in the column corresponding to the value in each row. So, for example, a record with a type of 'Sporty' would have a check in the 'Sporty' column. These new columns are known as '**dummy variables**'.

#### 4. Dummy variables: binary encoding

However, rather than having checks in the dummy variable columns it makes more sense to use binary values, where a one indicates the presence of the corresponding level. It might occur to you that the volume of data has exploded. You've gone from a single column of categorical values to six binary encoded dummy variables. If there were more levels then you'd have even more columns. This could get out of hand. However, the majority of the cells in the new columns contain zeros. The non-zero values, which actually encode the information, are relatively infrequent. This effect becomes even more pronounced if there are more levels. You can exploit this by converting the data into a sparse format.

#### 5. Dummy variables: sparse representation

Rather than recording the individual values, the **sparse representation** simply records the column numbers and value for the non-zero values.

#### 6. Dummy variables: redundant column

You can take this one step further. Since the categorical levels are mutually exclusive you can **drop one of the columns**. If type is not 'Midsize', 'Small', 'Compact', 'Sporty' or 'Large' then it must be 'Van'. **The process of creating dummy variables is called 'One-Hot Encoding'** because only one of the columns created is ever active or 'hot'. Let's see how this is done in Spark.

#### 7. One-hot encoding

As you might expect, there's a class for doing one-hot encoding. Import the OneHotEncoderEstimator class from the feature sub-module. When instantiating the class you need to specify the names of the input and output columns. For car type the input column is the index we defined earlier. Choose 'type_dummy' as the output column name. Note that these arguments are given as lists, so it's possible to specify multiple columns if necessary. Next fit the encoder to the data. Check how many category levels have been identified: six as expected.

#### 8. One-hot encoding

Now that the encoder is set up it can be applied to the data by calling the transform() method. Let's take a look at the results. There's now a type_dummy column which captures the dummy variables. As mentioned earlier, the final level is treated differently. **No column is assigned to type Van because if a vehicle isn't one of the other types then it must be a Van.** To have a separate dummy variable for Van would be redundant. The sparse format used to represent dummy variables looks a little complicated. Let's take a moment to dig into dense versus sparse formats.

#### 9. Dense versus **sparse**

**Suppose that you want to store a vector which consists mostly of zeros. You could store it as a dense vector, in which each of the elements of the vector is stored explicitly.** This is wasteful though because most of those elements are zeros. **A sparse representation is a much better alternative**. To create a sparse vector you need to specify the size of the vector (in this case, eight), the positions which are non-zero (in this case, positions zero and five, noting that we start counting at zero) and the values for each of those positions, one and seven. **Sparse representation is essential for effective one-hot encoding on large data sets.**

#### 10. One-Hot Encode categoricals

Let's try out one-hot encoding on the flights data.



### Encoding flight origin

The `org` column in the flights data is a categorical variable giving the airport from which a flight departs.

- ORD — O'Hare International Airport (Chicago)
- SFO — San Francisco International Airport
- JFK — John F Kennedy International Airport (New York)
- LGA — La Guardia Airport (New York)
- SMF — Sacramento
- SJC — San Jose
- TUS — Tucson International Airport
- OGG — Kahului (Hawaii)



The data are in a variable called `flights`. You have already used a string indexer to create a column of indexed values corresponding to the strings in `org`.



```python

# Import the one hot encoder class
from pyspark.ml.feature import OneHotEncoderEstimator

# Create an instance of the one hot encoder
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights_onehot = onehot.transform(flights)

# Check the results
flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()

```



### Regression

#### 1. Regression

In the previous lesson you learned how to one-hot encode categorical features, which is essential for building regression models. In this lesson you'll find out **how to build a regression model to predict numerical values.**

#### 2. Consumption versus mass: scatter

Returning to the cars data, suppose you wanted to predict fuel consumption using vehicle mass. **A scatter plot is a good way to visualize the relationship between those two variables.** Only a subset of the data are included in this plot, but it's clear that consumption increases with mass. However the relationship is not perfectly linear: there's scatter for individual points. A model should describe the average relationship of consumption to mass, without necessarily passing through individual points.

#### 3. Consumption versus mass: fit

This line, for example, might describe the underlying trend in the data.

#### 4. Consumption versus mass: alternative fits

But there are other lines which could equally well describe that trend. How do you choose the line which best describes the relationship?

#### 5. Consumption versus mass: residuals

First we need to define the concept of residuals. **The residual is the difference between the observed value and the corresponding modeled value.** The residuals are indicated in the plot as the vertical lines between the data points and the model line. The best model would somehow **make these residuals as small as possible.**

#### 6. Loss function

Out of all possible models, the best model is found by **minimizing a loss function**, which is an equation that describes how well the model fits the data. This is the equation for the mean squared error loss function. Let's quickly break it down.

#### 7. Loss function: Observed values

You've got the observed values, y_i, …

#### 8. Loss function: Model values

and the modeled values, \hat{y}_i. The difference between these is the residual. The residuals are squared and then summed together…

#### 9. Loss function: Mean

before finally dividing through by the number of data points to give the mean or average. **By minimizing the loss function you are effectively minimizing the average residual or the average distance between the observed and modeled values.** If this looks a little complicated, don't worry: Spark will do all of the maths for you.

$$\mathrm{MSE}=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}$$

#### 10. Assemble predictors

Let's build a regression model to predict fuel consumption using three predictors: mass, number of cylinders and vehicle type, where the last is a categorical which we've already one-hot encoded. **As before the first step towards building a model is to take our predictors and assemble them into a single column called 'features'.** The data are then randomly split into training and testing sets.

#### 11. Build regression model

The model is created using the LinearRegression class which is imported from the regression module. **By default this class expects to find the target data in a column called "label".** Since you are aiming to predict the "consumption" column you need to explicitly specify the name of the label column when creating a regression object. Next train the model on the training data using the fit() method. The trained model can then be used to making predictions on the testing data using the transform() method.

#### 12. Examine predictions

Comparing the predicted values to the known values from the testing data you'll see that there is reasonable agreement. It's hard to tell from a table though. A plot gives a clearer picture. The dashed diagonal lie represents perfect prediction. Most of the points lie close to this line, which is good.

#### 13. Calculate RMSE

It's useful to have a single number which summarizes the performance of a model. For classifiers there are a variety of such metrics. **The Root Mean Squared Error is often used for regression models**. It's the square root of the Mean Squared Error, which you've already encountered, and corresponds to the standard deviation of the residuals. The metrics for a classifier, like accuracy, precision and recall, are measured on an absolute scale where it's possible to immediately identify values that are "good" or "bad". Values of RMSE are relative to the scale of the value that you're aiming to predict, so interpretation is a little more challenging. A smaller RMSE, however, always indicates better predictions.

```python
from pyspark.ml.evaluation import RegressionEvaluator
# Find RMSE (Root Mean Squared Error)
RegressionEvaluator(labelCol='consumption').evaluate(predictions)
```

A RegressionEvaluator can also calculate the following metrics:
mae (Mean Absolute Error)
r2 (R )
mse (Mean Squared Error).

$$\mathrm{RMSD}=\sqrt{\frac{\sum_{i=1}^{N}\left(x_{i}-\hat{x}_{i}\right)^{2}}{N}}$$



#### 14. Consumption versus mass: intercept

Let's examine the model. The intercept is the value predicted by the model when all predictors are zero. On the plot this is the point where the model line intersects the vertical dashed line.

```python
regression.intercept
```

#### 15. Examine intercept

You can find this value for the model using the intercept attribute. This is the predicted fuel consumption when both mass and number of cylinders are zero and the vehicle type is 'Van'. Of course, this is an entirely hypothetical scenario: no vehicle could have zero mass!

#### 16. Consumption versus mass: slope

There's a slope associated with each of the predictors too, **which represents how rapidly the model changes when that predictor changes.**

#### 17. Examine Coefficients

The coefficients attribute gives you access to those values. There's a coefficient for each of the predictors. The coefficients for mass and number of cylinders are positive, indicating that heavier cars with more cylinders consume more fuel. These coefficients also represent the rate of change for the corresponding predictor. For example, the coefficient for mass indicates the change in fuel consumption when mass increases by one unit. Remember that there's no dummy variable for Van? The coefficients for the type dummy variables are relative to Vans. These coefficients should also be interpreted with care: if you are going to compare the values for different vehicle types then this needs to be done for fixed mass and number of cylinders. Since all of the type dummy coefficients are negative, the model indicates that, for a specific mass and number of cylinders, all other vehicle types consume less fuel than a Van. Large vehicles have the most negative coefficient, so it's possible to say that, for a specific mass and number of cylinders, Large vehicles are the most fuel efficient.

```Python
regression.coefficients
```



#### 18. Regression for numeric predictions

You've covered a lot of ground in this lesson. Let's apply what you've learned to the flights data.







### Flight duration model: Just distance


These are the features you'll include in the next model:

- `km`
- `org` (origin airport, one-hot encoded, 8 levels)
- `depart` (departure time, binned in 3 hour intervals, one-hot encoded, 8 levels)
- `dow` (departure day of week, one-hot encoded, 7 levels) and
- `mon` (departure month, one-hot encoded, 12 levels).

These have been assembled into the `features` column, which is a sparse representation of 32 columns (remember one-hot encoding produces a number of columns which is one fewer than the number of levels).

The data are available as `flights`, randomly split into `flights_train` and `flights_test`. The object `predictions` is also available.



```python

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select('duration', 'prediction').show(5, False)

# Calculate the RMSE
RegressionEvaluator(labelCol='duration').evaluate(predictions)

```



### Interpreting the coefficients

The linear regression model for flight duration as a function of distance takes the form

duration=α+β×distance

where

- α — intercept (component of duration which does not depend on distance) and
- β — coefficient (rate at which duration increases as a function of distance; also called the *slope*).

By looking at the coefficients of your model you will be able to infer

- how much of the average flight duration is actually spent on the ground and
- what the average speed is during a flight.

The linear regression model is available as `regression`.



```python

# Intercept (average minutes on ground)
inter = regression.intercept
print(inter)

# Coefficients
coefs = regression.coefficients
print(coefs)

# Average minutes per km
minutes_per_km = regression.coefficients[0]
print(minutes_per_km)

# Average speed in km per hour
avg_speed = 60 / minutes_per_km
print(avg_speed)

```



### Flight duration model: Adding origin airport

These data have been split into training and testing sets and are available as `flights_train` and `flights_test`. The origin airport, stored in the `org` column, has been indexed into `org_idx`, which in turn has been one-hot encoded into `org_dummy`. The first few records are displayed in the terminal.



```python

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
RegressionEvaluator(labelCol='duration').evaluate(predictions)

```





### Interpreting coefficients

Remember that origin airport, `org`, has eight possible values (ORD, SFO, JFK, LGA, SMF, SJC, TUS and OGG) which have been one-hot encoded to seven dummy variables in `org_dummy`.

The values for `km` and `org_dummy` have been assembled into `features`, which has eight columns with sparse representation. Column indices in `features` are as follows:

- 0 — `km`
- 1 — `ORD`
- 2 — `SFO`
- 3 — `JFK`
- 4 — `LGA`
- 5 — `SMF`
- 6 — `SJC` and
- 7 — `TUS`.

Note that `OGG` does not appear in this list because it is the reference level for the origin airport category.

In this exercise you'll be using the `intercept` and `coefficients` attributes to interpret the model.

The `coefficients` attribute is a list, where the first element indicates how flight duration changes with flight distance.

```python

# Average speed in km per hour
avg_speed_hour = 60 / regression.coefficients[0]
print(avg_speed_hour)

# Average minutes on ground at OGG
inter = regression.intercept
print(inter)

# Average minutes on ground at JFK
avg_ground_jfk = inter + regression.coefficients[3]
print(avg_ground_jfk)

# Average minutes on ground at LGA
avg_ground_lga = inter + regression.coefficients[4]
print(avg_ground_lga)

```



### Bucketing & Engineering

#### 1. Bucketing & Engineering

The largest improvements in Machine Learning model performance are often achieved by carefully **manipulating features**. In this lesson you'll be learning about a few approaches to doing this.

#### 2. **Bucketing**

Let's start with bucketing. It's often convenient to **convert a continuous variable**, like age or height, **into discrete values**. This can be done by assigning values to buckets or bins with well defined boundaries. **The buckets might have uniform or variable width.**

#### 3. Bucketing heights

Let's make this more concrete by thinking about observations of people's heights. If you plot the heights on a histogram then it seems reasonable…

#### 4. Bucketing heights

… to divide the heights up into ranges. To each of these ranges…

#### 5. Bucketing heights

… you assign a label. Then you create a new column in the data…

#### 6. Bucketing heights

… with the appropriate labels. The resulting categorical variable is often a more powerful predictor than the original continuous variable.

#### 7. RPM histogram

Let's apply this to the cars data. Looking at the distribution of values for RPM you see that the majority lie in the range between 4500 and 6000. There are a few either below or above this range. This suggests that it would make sense to bucket these values according to those boundaries.

#### 8. RPM buckets

You create a bucketizer object, specifying the bin boundaries as the "splits" argument and also providing the names of the input and output columns. You then apply this object to the data by calling the transform() method.

#### 9. RPM buckets

The result has a new column with the discrete bucket values. The three buckets have been assigned index values zero, one and two, corresponding to the low, medium and high ranges for RPM.

#### 10. One-hot encoded RPM buckets

As you saw earlier, before you can use these index values in a regression model, they first need to be one-hot encoded. The low and medium RPM ranges are mapped to distinct dummy variables, while the high range is the reference level and does not get a separate dummy variable.

#### 11. Model with bucketed RPM

Let's look at the intercept and coefficients for a model which predicts fuel consumption based on bucketed RPM data. The intercept tells us what the fuel consumption is for the reference level, which is the high RPM bucket. To get the consumption for the low RPM bucket you add the first coefficient to the intercept. Similarly, to find the consumption for the medium RPM bucket you add the second coefficient to the intercept.

#### 12. More feature engineering

There are many other approaches to engineering new features. It's common to apply arithmetic operations to one or more columns to create new features.

#### 13. Mass & Height to BMI

Returning to the heights data. Suppose that we also had data for mass.

#### 14. Mass & Height to BMI

Then it might be perfectly reasonable to engineer a new column for BMI. Potentially BMI might be a more powerful predictor than either height or mass in isolation.

#### 15. Engineering density

Let's apply this idea to the cars data. You have columns for mass and length. **Perhaps some combination of the two might be even more meaningful.** You can create different forms of density by dividing the mass through by the first three powers of length. Since you only have the length of the vehicles but not their width or height, the **length is being used as a proxy for these missing dimensions.** In so doing you create three new predictors. The first density represents how mass changes with vehicle length. The second and third densities approximate how mass varies with the area and volume of the vehicle. Which of these will be meaningful for our model? Right now you don't know, you're just trying things out. Powerful new features are often discovered through trial and error. In the next lesson you'll learn about a technique for selecting only the relevant predictors in a regression model.

#### 16. Let's engineer some features!

Right now though, let's apply what you've learned to the flights data.



### Bucketing departure time

Time of day data are a challenge with regression models. They are also a great candidate for bucketing.

In this lesson you will convert the flight departure times from numeric values between 0 (corresponding to 00:00) and 24 (corresponding to 24:00) to binned values. 



```python

from pyspark.ml.feature import Bucketizer, OneHotEncoderEstimator

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits=[0, 3, 6, 9, 12, 15, 18, 21, 24], inputCol='depart', outputCol='depart_bucket')

# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select('depart', 'depart_bucket').show(5)

# Create a one-hot encoder
onehot = OneHotEncoderEstimator(inputCols=['depart_bucket'], outputCols=['depart_dummy'])

# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)

```



### Flight duration model: Adding departure time

The data are in `flights`. The `km`, `org_dummy` and `depart_dummy` columns have been assembled into `features`, where `km` is index 0, `org_dummy` runs from index 1 to 7 and `depart_dummy` from index 8 to 14.

The data have been split into training and testing sets and a linear regression model, `regression`, has been built on the training data. Predictions have been made on the testing data and are available as `predictions`.



```python

# Find the RMSE on testing data
from pyspark.ml.evaluation import RegressionEvaluator
RegressionEvaluator(labelCol='duration').evaluate(predictions)

# Average minutes on ground at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 00:00 and 03:00
avg_night_ogg = regression.intercept + regression.coefficients[8]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 00:00 and 03:00
avg_night_jfk = regression.intercept + regression.coefficients[8] + regression.coefficients[3]
print(avg_night_jfk)

```



### Regularization

#### 1. Regularization

The regression models that you've built up until now have blindly included all of the provided features. Next you are going to learn about a more sophisticated model which effectively selects only the most useful features.

#### 2. Features: Only a few

A linear regression model attempts to derive a coefficient for each feature in the data. The coefficients quantify the effect of the corresponding features. More features imply more coefficients. This works well when your dataset has a few columns and many rows. You need to derive a few coefficients and you have plenty of data.

#### 3. Features: Too many

The converse situation, many columns and few rows, is much more challenging. Now you need to calculate values for numerous coefficients but you don't have much data to do it. Even if you do manage to derive values for all of those coefficients, your model will end up being very complicated and difficult to interpret. Ideally you want to create a parsimonious model: one that has just the minimum required number of predictors. It will be as simple as possible, yet still able to make robust predictions.

#### 4. Features: Selected

The obvious solution is to simply select the "best" subset of columns. But how to choose that subset? There are a variety of approaches to this "**feature selection**" problem.

#### 5. Loss function (revisited)

In this lesson we'll be exploring one such approach to feature selection known as "**penalized regression**". The basic idea is that the model is penalized, or punished, for having too many coefficients. Recall that the conventional regression algorithm chooses coefficients to minimize the loss function, which is **average of the squared residuals**. A good model will result in low MSE because its predictions will be close to the observed values.

$$\mathrm{MSE}=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}$$

#### 6. Loss function with regularization

With penalized regression an additional "**regularization**" or "**shrinkage**" term is added to the loss function. Rather than depending on the data, this term is a function of the model coefficients.

$$\mathrm{MSE}=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2}+\lambda f(\beta)$$

#### 7. Regularization term

There are **two standard forms** for the regularization term. **Lasso regression** uses a term which is proportional to the **absolute value of the coefficients,** while Ridge regression uses **the square of the coefficients.** In both cases this extra term in the loss function penalizes models with too many coefficients. There's a subtle distinction between Lasso and Ridge regression. Both will shrink the coefficients of unimportant predictors. **However, whereas Ridge will result in those coefficients being close to zero, Lasso will actually force them to zero precisely.** It's also possible to have a mix of Lasso and Ridge. The strength of the regularization is determined by a parameter which is generally denoted by the Greek symbol lambda. **When lambda = 0 there is no regularization and when lambda is large regularization completely dominates.** Ideally you want to choose a value for lambda between these two extremes!

#### 8. Cars again

Let's make this more concrete by returning to the cars data. We've assembled the mass, cylinders and type columns along with the freshly engineered density columns. We've effectively got ten predictors available for the model. As usual we'll split these data into training and testing sets.

#### 9. Cars: Linear regression

Let's start by fitting a standard linear regression model to the training data. You can then make predictions on the testing data and calculate the RMSE. When you look at the model coefficients you find that all predictors have been assigned non-zero values. This means that every predictor is contributing to the model. This is certainly possible, but it's unlikely that all of the features are actually important for predicting consumption.

```python
regression.coefficients

# DenseVector([-0.012, 0.174, -0.897, -1.445, -0.985, -1.071, -1.335, 0.189, -0.780, 1.160])
```



#### 10. Cars: Ridge regression

Now let's fit a Ridge Regression model to the same data. You get a Ridge Regression model by **giving a value of zero for elasticNetParam**. An **arbitrary value of 0.1** has been chosen for the regularization strength. Later you'll learn a way to choose good values for this parameter based on the data. When you calculate the RMSE on the testing data you find that it has increased slightly, but not enough to cause concern. Looking at the coefficients you see that they are all smaller than the coefficients for the standard linear regression model. They have been "shrunk".

$$\mathrm{RMSD}=\sqrt{\frac{\sum_{i=1}^{N}\left(x_{i}-\hat{x}_{i}\right)^{2}}{N}}$$

#### 11. Cars: Lasso regression

Finally **let's build a Lasso Regression model, by setting elasticNetParam to 1.** Again **you find that the testing RMSE has increased, but not by a significant degree.** Turning to the coefficients though, you see that something important has happened: all but two of the coefficients are now zero. There are effectively only two predictors left in the model: the dummy variable for a small type car and the linear density. Lasso Regression has identified the most important predictors and set the coefficients for the rest to zero. This tells us that we can get a good model by simply knowing whether or not a car is 'small' and it's linear density. A simpler model with no significant loss in performance.

#### 12. Regularization ? simple model

Let's try out regularization on our flight duration model.



### Flight duration model: More features!

More features will *always* make the model more complicated and difficult to interpret.

- `km`
- `org` (origin airport, one-hot encoded, 8 levels)
- `depart` (departure time, binned in 3 hour intervals, one-hot encoded, 8 levels)
- `dow` (departure day of week, one-hot encoded, 7 levels) and
- `mon` (departure month, one-hot encoded, 12 levels).



These have been assembled into the `features` column, which is a sparse representation of 32 columns (remember one-hot encoding produces a number of columns which is one fewer than the number of levels).

The data are available as `flights`, randomly split into `flights_train` and `flights_test`. The object `predictions` is also available.



```python

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit linear regression model to training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Make predictions on testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

```



### Flight duration model: Regularisation!

In this exercise you'll use Lasso regression (regularized with a L1 penalty) to create a more parsimonious model. Many of the coefficients in the resulting model will be set to zero. This means that only a subset of the predictors actually contribute to the model. Despite the simpler model, it still produces a good RMSE on the testing data.

```python

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit Lasso model (α = 1) to training data
regression = LinearRegression(labelCol='duration', regParam=1, elasticNetParam=1).fit(flights_train)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(regression.transform(flights_test))
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# Number of zero coefficients
zero_coeff = sum([beta == 0 for beta in regression.coefficients])
print("Number of coefficients equal to 0:", zero_coeff)

```



## 4. Ensembles & Pipelines

### Pipeline

#### 1. Pipeline

Welcome back! So far you've learned how to build classifier and regression models using Spark. In this chapter you'll learn how to make those models better. You'll start by taking a look at **pipelines, which will seriously streamline your workflow**. They will also help to ensure that training and testing data are treated consistently and that no leakage of information between these two sets takes place.

#### 2. Leakage?

What do I mean by leakage? Most of the actions you've been using involve both a fit() and a transform() method. Those methods have been applied in a fairly relaxed way. But to get really robust results you need to be careful only to apply the fit() method to training data. Why? Because if a fit() method is applied to *any* of the testing data then the model will effectively have seen those data during the training phase, so the results of testing will no longer be objective. The transform() method, on the other hand, can be applied to both training and testing data since it does not result in any changes in the underlying model.

#### 3. A leaky model

A figure should make this clearer. **Leakage occurs whenever a fit() method is applied to testing data.** Suppose that you fit a model using both the training and testing data. The model would then already have *seen* the testing data, so using those data to test the model would not be fair: of course the model will perform well on data which has been used for training! This sounds obvious, but care must be taken not to fall into this trap. Remember that there are normally multiple stages in building a model and if the fit() method in *any* of those stages is applied to the testing data then the model is compromised.

#### 4. A watertight model

However, if you are careful to only apply fit() to the training data then your model will be in good shape. When it comes to testing it will not have seen *any* of the testing data and the test results will be completely objective. **Luckily a pipeline will make it easier to avoid leakage because it simplifies the training and testing process.**

#### 5. Pipeline

**A pipeline is a mechanism to combine a series of steps.** Rather than applying each of the steps individually, they are all grouped together and applied as a single unit.

#### 6. Cars model: Steps

Let's return to our cars regression model. Recall that there were a number of steps involved: - using a string indexer to convert the type column to indexed values; - applying a one-hot encoder to convert those indexed values into dummy variables; then - assembling a set of predictors into a single features column; and finally - building a regression model.

#### 7. Cars model: Applying steps

Let's map out the process of applying those steps. - **First** you fit the **indexer** to the training data. Then you call the transform() method on the training data to add the indexed column. - Then you call the transform() method on the testing data to add the indexed column there too. Note that the testing data was not used to fit the indexer. **Next** you do the same things for the **one-hot encoder**, fitting to the training data and then using the fitted encoder to update the training and testing data sets. The assembler is next. In this case there is no fit() method, so you simply apply the transform() method to the training and testing data. **Finally** the data are ready. You fit the **regression model** to the training data and then use the model to make predictions on the testing data. Throughout the process you've been careful to keep the testing data out of the training process. But this is hard work and it's easy enough to slip up.

#### 8. Cars model: Pipeline

A pipeline makes training and testing a complicated model a lot easier. The Pipeline class lives in the ml sub-module. You create a pipeline by specifying a sequence of stages, where each stage corresponds to a step in the model building process. The stages are executed in order. Now, rather than calling the fit() and transform() methods for each stage, you simply call the fit() method for the pipeline on the training data. Each of the stages in the pipeline is then automatically applied to the training data in turn. This will systematically apply the fit() and transform() methods for each stage in the pipeline. The trained pipeline can then be used to make predictions on the testing data by calling its transform() method. The pipeline transform() method will only call the transform() method for each of the stages in the pipeline. Isn't that simple?

#### 9. Cars model: Stages

You can access the stages in the pipeline by using the .stages attribute, which is a list. You pick out individual stages by indexing into the list. For example, to access the regression component of the pipeline you'd use an index of 3. Having access to that component makes it possible to get the intercept and coefficients for the trained LinearRegression model.

#### 10. Pipelines streamline workflow!

Pipelines make your code easier to read and maintain. Let's try them out with our flights model.



### Flight duration model: Pipeline stages

```python

# Convert categorical strings to index values
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# One-hot encode index values
onehot = OneHotEncoderEstimator(
    inputCols=['org_idx', 'dow'],
    outputCols=['org_dummy', 'dow_dummy']
)

# Assemble predictors into a single column
assembler = VectorAssembler(inputCols=['km', 'org_dummy', 'dow_dummy'], outputCol='features')

# A linear regression object
regression = LinearRegression(labelCol='duration')

```



### Flight duration model: Pipeline model

The data are available as `flights`, which has been randomly split into `flights_train` and `flights_test`.

```python

# Import class for creating a pipeline
from pyspark.ml import Pipeline

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)

# Make predictions on the testing data
predictions = pipeline.transform(flights_test)

```



### SMS spam pipeline



You haven't looked at the SMS data for quite a while. Last time we did the following:

- split the text into tokens
- removed stop words
- applied the hashing trick
- converted the data from counts to IDF and
- trained a logistic regression model.

Each of these steps was done independently. This seems like a great application for a pipeline!



```python

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol='text', outputCol='words')

# Remove stop words
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol=remover.getOutputCol(), outputCol="hash")
idf = IDF(inputCol=hasher.getOutputCol(), outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])

```





#### Cross-Validation

#### 1. Cross-Validation

Up until now you've been testing models using a rather simple technique: randomly splitting the data into training and testing sets, training the model on the training data and then evaluating its performance on the testing set. **There's one major drawback to this approach: you only get one estimate of the model performance**. You would have a more robust idea of how well a model works if you were able to test it multiple times. This is precisely the idea behind cross-validation.

#### 2. CV - complete data

You start out with the full set of data.

#### 3. CV - train/test split

You still split these data into a training set and a testing set. Remember that before splitting it's important to **first randomize the data** so that the distributions in the training and testing data are similar.

#### 4. CV - multiple folds

You then **split the training data into a number of partitions or "folds".** The number of folds normally factors into the name of the technique. For example, if you split into five folds then you'd talk about 5-fold cross-validation.

#### 5. Fold upon fold - first fold

Once the training data have been split into folds you can start cross-validating. First keep aside the data in the first fold. Train a model on the remaining four folds. **Then evaluate that model on the data from the first fold. This will give the first value for the evaluation metric.**

#### 6. Fold upon fold - second fold

Next you move onto the second fold, where the same process is repeated: data in the second fold are set aside for testing while the remaining four folds are used to train a model. That model is tested on the second fold data, yielding the second value for the evaluation metric.

#### 7. Fold upon fold - other folds

You repeat the process for the remaining folds. Each of the folds is used in turn as testing data and you end up with as many values for the evaluation metric as there are folds. At this point you are in a position to calculate the average of the evaluation metric over all folds, which is a much more robust measure of model performance than a single value.

#### 8. Cars revisited

Let's see how this works in practice. Remember the cars data? Of course you do. You're going to build a cross-validated regression model to predict consumption.

#### 9. Estimator and evaluator

Here are the first two ingredients which you need to perform cross-validation: - an estimator, which builds the model and is often a pipeline; and - an evaluator, which quantifies how well a model works on testing data. We've seen both of these a few times already.

#### 10. Grid and cross-validator

Now the final ingredients. You'll need two new classes, **CrossValidator** and **ParamGridBuilder**, both from the tuning sub-module. You'll create a parameter grid, which you'll leave empty for the moment, but will return to in detail during the next lesson. Finally you have everything required to create a cross-validator object: - an estimator, which is the linear regression model, - an empty grid of parameters for the estimator and - an evaluator which will calculate the RMSE. You can optionally specify the number of folds (which defaults to three) and a random number seed for repeatability.

#### 11. Cross-validators need training too

The cross-validator has a fit() method which will apply the cross-validation procedure to the training data. You can then look at the average RMSE calculated across all of the folds. This is a more robust measure of model performance because it is based on multiple train/test splits. Note that the average metric is returned as a list. You'll see why in the next lesson.

#### 12. Cross-validators act like models

The trained cross-validator object acts just like any other model. It has a transform method, which can be used to make predictions on new data. If we evaluate the predictions on the original testing data then we find a smaller value for the RMSE than we obtained using cross-validation. This means that a simple train-test split would have given an overly optimistic view on model performance.

#### 13. Cross-validate all the models!

Let's give cross-validation a try on our flights model.



### Cross validating simple flight duration model

The data have been randomly split into `flights_train` and `flights_test`.

The following classes have already been imported: `LinearRegression`, `RegressionEvaluator`, `ParamGridBuilder` and `CrossValidator`.



```python

# import classes
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol='duration')
evaluator = RegressionEvaluator(labelCol='duration')

# Create a cross validator
cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# NOTE: Since cross-valdiation builds multiple models, the fit() method can take a little while to complete.

```





### Cross validating flight duration model pipeline



The following objects have already been created:

- `params` — an empty parameter grid
- `evaluator` — a regression evaluator
- `regression` — a `LinearRegression` object with `labelCol='duration'`.

All of the required classes have already been imported.



```python

# Create an indexer for the org field
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoderEstimator(inputCols=['org_idx'], outputCols=['org_dummy'])

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols=['km', 'org_dummy'], outputCol='features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params,
                    evaluator=evaluator)

```



### Grid Search

#### 1. Grid Search

So far you've been using the default parameters for almost everything. You've built some decent models, but they could probably be improved by choosing better model parameters.

#### 2. Tuning

There is no universal "best" set of parameters for a particular model. The optimal choice of parameters will **depend on the data and the modeling goal.** The idea is relatively simple, you build a selection of models, one for each set of model parameters. Then you evaluate those models and choose the best one.

#### 3. Cars revisited (again)

You'll be looking at the fuel consumption regression model again.

#### 4. Fuel consumption with intercept

You'll start by doing something simple, comparing a linear regression model with an intercept to one that passes through the origin. By default a linear regression model will always fit an intercept, but you're going to be explicit and specify the fitIntercept parameter as True. You fit the model to the training data and then calculate the RMSE for the testing data.

#### 5. Fuel consumption without intercept

Next you repeat the process, but specify False for the fitIntercept parameter. Now you are creating a model which passes through the origin. When you evaluate this model you find that the RMSE is higher. So, comparing these two models you'd naturally choose the first one because it has a lower RMSE. However, there's a problem with this approach. Just getting a single estimate of RMSE is not very robust. It'd be better to make this comparison using cross-validation. You also have to manually build the models for the two different parameter values. It'd be great if that were automated.

#### 6. Parameter grid

You can systematically evaluate a model across a grid of parameter values using a technique known as grid search. To do this you need to set up a parameter grid. You actually saw this in the previous lesson, where you simply created an empty grid. Now you are going to add points to the grid. First you create a grid builder and then you add one or more grids. At present there's just one grid, which takes two values for the fitIntercept parameter. Call the build() method to construct the grid. A separate model will be built for each point in the grid. You can check how many models this corresponds to and, of course, this is just two.

#### 7. Grid search with cross-validation

Now you create a cross-validator object and fit it to the training data. This builds a bunch of models: one model for each fold and point in the parameter grid. Since there are two points in the grid and ten folds, this translates into twenty models. The cross-validator is going to loop through each of the points in the parameter grid and for each point it will create a cross-validated model using the corresponding parameter values. When you take a look at the average metrics attribute, you can see why the metric is given as a list: you get one average value for each point in the grid. The values confirm what you observed before: the model that includes an intercept is superior to the model without an intercept.

#### 8. The best model & parameters

Our goal was to get the best model for the data. You retrieve this using the appropriately named bestModel attribute. But it's not actually necessary to work with this directly because the cross-validator object will behave like the best model. So, you can use it directly to make predictions on the testing data. Of course, you want to know what the best parameter value is and you can retrieve this using the explainParam() method. As expected the best value for the fitIntercept parameter is True. You can see this after the word "current" in the output.

#### 9. A more complicated grid

It's possible to add more parameters to the grid. Here, in addition to whether or not to include an intercept, you're also considering a selection of values for the regularization parameter and the elastic net parameter. Of course, the more parameters and values you add to the grid, the more models you have to evaluate. Because each of these models will be evaluated using cross-validation, this might take a little while. But it will be time well spent, because the model that you get back will in principle be much better than what you would have obtained by just using the default parameters.

#### 10. Find the best parameters!

Let's apply grid search on the flights and SMS models!



### Optimizing flights linear regression

The following have already been created:

- `regression` — a `LinearRegression` object
- `pipeline` — a pipeline with string indexer, one-hot encoder, vector assembler and linear regression and
- `evaluator` — a `RegressionEvaluator` object.

```python

# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0]) \
               .addGrid(regression.elasticNetParam, [0.0, 0.5, 1.0])

# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params))

# Create cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

```

### Dissecting the best flight duration model

The following have already been created:

- `cv` — a trained `CrossValidatorModel` object and
- `evaluator` — a `RegressionEvaluator` object.

The flights data have been randomly split into `flights_train` and `flights_test`.

```python

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
evaluator.evaluate(predictions)

```



### SMS spam optimised

The following are already defined:

- `hasher` — a `HashingTF` object and
- `logistic` — a `LogisticRegression` object.

```python

# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, [1024, 4096, 16384]) \
               .addGrid(hasher.binary, [True, False])

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]) \
               .addGrid(logistic.elasticNetParam, [0.0, 0.5, 1.0])

# Build parameter grid
params = params.build()

```



### How many models for grid search?

How many models will be built when the cross-validator below is fit to data?

```python
params = ParamGridBuilder().addGrid(hasher.numFeatures, [1024, 4096, 16384]) \
                           .addGrid(hasher.binary, [True, False]) \
                           .addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]) \
                           .addGrid(logistic.elasticNetParam, [0.0, 0.5, 1.0]) \
                           .build()

cv = CrossValidator(..., estimatorParamMaps=params, numFolds=5)
```

There are 72 points in the parameter grid and 5 folds in the cross-validator. The product is **360**.



### Ensemble

#### 1. Ensemble

You now know how to choose a good set of parameters for any model using cross-validation and grid search. In the final lesson you're going to learn about how models can be combined to form a collection or "ensemble" which is more powerful than each of the individual models alone.

#### 2. What's an ensemble?

Simply put, **an ensemble model is just a collection of models.** **An ensemble model combines the results from multiple models to produce better predictions than any one of those models acting alone.** The concept is based on the idea of the "Wisdom of the Crowd", which implies that the aggregated opinion of a group is better than the opinions of the individuals in that group, even if the individuals are experts.

#### 3. Ensemble diversity

As the quote suggests, for this idea to be true, there must be **diversity and independence in the crowd.** This applies to models too: a successful ensemble requires diverse models. **It does not help if all of the models in the ensemble are similar or exactly the same.** Ideally each of the models in the ensemble should be different.

#### 4. Random Forest

**A Random Forest, as the name implies, is a collection of trees.** To ensure that each of those trees is different, the Decision Tree algorithm is modified slightly: - **each tree is trained on a different random subset of the data** and - within each tree a random subset of features is used for splitting at each node. The result is a collection of trees where no two trees are the same. Within the Random Forest model, all of the trees operate in parallel.

#### 5. Create a forest of trees

Let's go back to the cars classifier yet again. You create a Random Forest model using the RandomForestClassifier class from the classification sub-module. You can select the number of trees in the forest using the numTrees parameter. By default this is twenty, but we'll drop that to five so that the results are easier to interpret. As is the case with any other model, the Random Forest is fit to the training data.

```python
# Returning to cars data: manufactured in USA ( 0.0 ) or not ( 1.0 ).
# Create Random Forest classi
from pyspark.ml.classification import RandomForestClassifier
forest = RandomForestClassifier(numTrees=5)
# Fit to the training data
forest = forest.fit(cars_train)
```

#### 6. Seeing the trees

Once the model is trained it's possible to access the individual trees in the forest using the trees attribute. You would not normally do this, but it's useful for illustrative purposes. There are precisely five trees in the forest, as specified. The trees are all different, as can be seen from the varying number of nodes in each tree. You can then make predictions using each tree individually.

```python
forest.trees
```

#### 7. Predictions from individual trees

Here are the predictions of individual trees on a subset of the testing data. Each row represents predictions from each of the five trees for a specific record. In some cases all of the trees agree, but there is often some dissent amongst the models. This is precisely where the Random Forest works best: where the prediction is not clear cut. The Random Forest model creates a consensus prediction by aggregating the predictions across all of the individual trees.

#### 8. Consensus predictions

You don't need to worry about these details though because the transform() method will automatically generate a consensus prediction column. It also creates a probability column which assigns aggregate probabilities to each of the outcomes.

#### 9. Feature importances

It's possible to get an idea of the relative importance of the features in the model by looking at the featureImportances attribute. An importance is assigned to each feature, where a larger importance indicates a feature which makes a larger contribution to the model. Looking carefully at the importances we see that feature 4 (rpm) is the most important, while feature 0 (the number of cylinders) is the least important.

```python
forest.featureImportances
# SparseVector(6, {0: 0.0205, 1: 0.2701, 2: 0.108, 3: 0.1895, 4: 0.2939, 5: 0.1181})

```

`rpm` is most important `cyl` is least important.

#### 10. Gradient-Boosted Trees

**The second ensemble model you'll be looking at is Gradient-Boosted Trees.** Again the aim is to build a collection of diverse models, but the approach is slightly different. **Rather than building a set of trees that operate in parallel, now we build trees which work in series.** The boosting algorithm works iteratively. First build a decision tree and add to the ensemble. Then use the ensemble to make predictions on the training data. Compare the predicted labels to the known labels. Now identify training instances where predictions were incorrect. Return to the start and train another tree which focuses on improving the incorrect predictions. As trees are added to the ensemble its predictions improve because each new tree focuses on correcting the shortcomings of the preceding trees.

#### 11. Boosting trees

The class for the Gradient-Boosted Tree classifier is also found in the classification sub-module. After creating an instance of the class you fit it to the training data.

```python
# Create a Gradient-Boosted Tree classi
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
# Fit to the training data.
gbt = gbt.fit(cars_train)
```

#### 12. Comparing trees

You can make an objective comparison between a plain Decision Tree and the two ensemble models by looking at the values of AUC obtained by each of them on the testing data. Both of the ensemble methods score better than the Decision Tree. This is not too surprising since they are significantly more powerful models. It's also worth noting that these results are based on the default parameters for these models. It should be possible to get even better performance by tuning those parameters using cross-validation.

```python
# AUC for Decision Tree
0.5875
# AUC for Random Forest
0.65
# AUC for Gradient-Boosted Tree
0.65
```

#### 13. Ensemble all of the models!

In the final set of exercises you'll try out ensemble methods on the flights data.



### Delayed flights with Gradient-Boosted Trees

You've previously built a classifier for flights likely to be delayed using a Decision Tree. In this exercise you'll compare a Decision Tree model to a Gradient-Boosted Trees model.

The flights data have been randomly split into `flights_train` and `flights_test`.



```python

# Import the classes required
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(tree.transform(flights_test))
evaluator.evaluate(gbt.transform(flights_test))

# Find the number of trees and the relative importance of features
print(gbt.getNumTrees)
print(gbt.featureImportances)

```



### Delayed flights with a Random Forest

You'll be training a Random Forest classifier to predict delayed flights, using cross validation to choose the best values for model parameters.

You'll find good values for the following parameters:

- `featureSubsetStrategy` — the number of features to consider for splitting at each node and
- `maxDepth` — the maximum number of splits along any branch.

Unfortunately building this model takes too long, so we won't be running the `.fit()` method on the pipeline

```python

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder
# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder() \
            .addGrid(forest.featureSubsetStrategy, ['all', 'onethird', 'sqrt', 'log2']) \
            .addGrid(forest.maxDepth, [2, 5, 10]) \
            .build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(estimator=forest, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

```



### Evaluating Random Forest

In this final exercise you'll be evaluating the results of cross-validation on a Random Forest model.

The following have already been created:

- `cv` - a cross-validator which has already been fit to the training data
- `evaluator` — a `BinaryClassificationEvaluator` object and
- `flights_test` — the testing data.



```python

# Average AUC for each parameter combination in grid
avg_auc = cv.avgMetrics

# Average AUC for the best model
best_model_auc = max(cv.avgMetrics)

# What's the optimal parameter value?
opt_max_depth = cv.bestModel.explainParam('maxDepth')
opt_feat_substrat = cv.bestModel.explainParam('featureSubsetStrategy')

# AUC for best model on testing data
best_auc = evaluator.evaluate(cv.transform(flights_test))

```



### Closing thoughts

#### 1. Closing thoughts

Congratulations on completing this course on Machine Learning with Apache Spark. You have covered a lot of ground, reviewing some Machine Learning fundamentals and seeing how they can be applied to large datasets, using Spark for distributed computing.

#### 2. Things you've learned

You learned how to load data into Spark and then perform a variety of operations on those data. Specifically, you learned basic column manipulation on DataFrames, how to deal with text data, bucketing continuous data and one-hot encoding categorical data. You then delved into two types of classifiers, Decision Trees and Logistic Regression, in the process building a robust spam classifier. You also learned about partitioning your data and how to use testing data and a selection of metrics to evaluate a model. Next you learned about regression, starting with a simple linear regression model and progressing to penalized regression, which allowed you to build a model using only the most relevant predictors. You learned about pipelines and how they can make your Spark code cleaner and easier to maintain. This led naturally into using cross-validation and grid search to derive more robust model metrics and use them to select good model parameters. Finally you encountered two forms of ensemble models.

#### 3. Learning more

Of course, there are many topics that were not covered in this course. If you want to dig deeper then consult the excellent and extensive online documentation. Importantly you can find instructions for setting up and securing a Spark cluster.

#### 4. Congratulations!

Now go and use what you've learned to solve challenging and interesting big data problems in the real world!

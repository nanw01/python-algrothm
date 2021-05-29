# Cleaning Data with PySpark

## DataFrame details

### Defining a schema

#### 1. Intro to data cleaning with Apache Spark

Welcome to Data Cleaning in Apache Spark with Python. My name is Mike Metzger, I am a Data Engineering Consultant, and I will be your instructor for this course. We will cover what data cleaning is, why it's important, and how to implement it with Spark and Python. Let's get started!

#### 2. What is Data Cleaning?

In this course, we'll define "data cleaning" as preparing raw data for use in processing pipelines. We'll discuss what a pipeline is later on, but for now, it's sufficient to say that data cleaning is a necessary part of any production data system. If your data isn't "clean", it's not trustworthy and could cause problems later on. There are many tasks that could fall under the data cleaning umbrella. A few of these include reformatting or replacing text; performing calculations based on the data; and removing garbage or incomplete data.

#### 3. Why perform data cleaning with Spark?

Most data cleaning systems have two big problems: optimizing performance and organizing the flow of data. A typical programming language (such as Perl, C++, or even standard SQL) may be able to clean data when you have small quantities of data. But consider what happens when you have millions or even billions of pieces of data. Those languages wouldn't be able to process that amount of information in a timely manner. Spark lets you scale your data processing capacity as your requirements evolve. Beyond the performance issues, dealing with large quantities of data requires a process, or pipeline of steps. Spark allows management of many complex tasks within a single framework.

#### 4. Data cleaning example

Here's an example of cleaning a small data set. We're given a table of names, age in years, and a city. Our requirements are for a DataFrame with first and last name in separate columns, the age in months, and which state the city is in. We also want to remove any rows where the data is out of the ordinary. Using Spark transformations, we can create a DataFrame with these properties and continue processing afterwards.

#### 5. Spark Schemas

A primary function of data cleaning is to verify all data is in the expected format. Spark provides a built-in ability to validate datasets with schemas. You may have used schemas before with databases or XML; Spark is similar. A schema defines and validates the number and types of columns for a given DataFrame. A schema can contain many different types of fields - integers, floats, dates, strings, and even arrays or mapping structures. A defined schema allows Spark to filter out data that doesn't conform during read, ensuring expected correctness. In addition, schemas also have performance benefits. Normally a data import will try to infer a schema on read - this requires reading the data twice. Defining a schema limits this to a single read operation.

#### 6. Example Spark Schema

Here is an example schema to the import data from our previous example. First we'll import the pyspark.sql.types library. Next we define the actual StructType list of StructFields, containing an entry for each field in the data. Each StructField consists of a field name, dataType, and whether the data can be null. Once our schema is defined, we can add it into our spark.read.format.load call and process it against our data. The load() method takes two arguments - the filename and a schema. This is where we apply our schema to the data being loaded.

#### 7. Let's practice!

We've gone over a lot of information regarding data cleaning and the importance of dataframe schemas. Let's put that information to use and practice!



### Data cleaning review

There are many benefits for using Spark for data cleaning.



Which of the following is *NOT* a benefit?

- Spark offers high performance.

- Spark allows orderly data flows.

- Spark can use strictly defined schemas while ingesting data.

  ### Defining a schema

Creating a defined schema helps with data quality and import performance. As mentioned during the lesson, we'll create a simple schema to read in the following columns:

- Name
- Age
- City

The `Name` and `City` columns are `StringType()` and the `Age` column is an `IntegerType()`.

```python
# Import the pyspark.sql.types library
from pyspark.sql.types import *

# Define a new schema using the StructType method
people_schema = StructType([
  # Define a StructField for each field
  StructField('name', StringType(), False),
  StructField('age', IntegerType(), True),
  StructField('city', StringType(), True)
])
```



### Immutability and lazy processing



#### 1. Immutability and Lazy Processing

Welcome back! We've had a quick discussion about data cleaning, data types and schemas. Let's move on to some further Spark concepts - Immutability and Lazy Processing.

#### 2. Variable review

Normally in Python, and most other languages, variables are fully mutable. The values can be changed at any given time, assuming the scope of the variable is valid. While very flexible, this does present problems anytime there are multiple concurrent components trying to modify the same data. Most languages work around these issues using constructs like mutexes, semaphores, etc. This can add complexity, especially with non-trivial programs.

#### 3. Immutability

Unlike typical Python variables, Spark Data Frames are immutable. While not strictly required, immutability is often a component of functional programming. We won't go into everything that implies here, but understand that Spark is designed to use immutable objects. Practically, **this means Spark Data Frames are defined once and are not modifiable after initialization.** If the variable name is reused, the original data is removed (assuming it's not in use elsewhere) and the variable name is reassigned to the new data. While this seems inefficient, it actually allows Spark to share data between all cluster components. It can do so without worry about concurrent data objects.

#### 4. Immutability Example

This is a quick example of the immutability of data frames in Spark. It's OK if you don't understand the actual code, this example is more about the concepts of what happens. First, we create a data frame from a CSV file called voterdata.csv. This creates a new data frame definition and assigns it to the variable name voter_df. Once created, we want to do two further operations. The first is to create a fullyear column by using a 2-digit year present in the data set and adding 2000 to each entry. This does not actually change the data frame at all. It copies the original definition, adds the transformation, and assigns it to the voter_df variable name. Our second operation is similar - now we want to drop the original year column from the data frame. Again, this copies the definition, adds a transformation and reassigns the variable name to this new object. The original objects are destroyed. Please note that the original year column is now permanently gone from this instance, though not from the underlying data (ie, you could simply reload it to a new dataframe if desired).

#### 5. Lazy Processing

You may be wondering how Spark does this so quickly, especially on large data sets. Spark can do this because of something called **lazy processing.** Lazy processing in Spark is the idea that very little actually happens until an action is performed. In our previous example, we read a CSV file, added a new column, and deleted another. The trick is that no data was actually read / added / modified, we only updated the instructions (aka, Transformations) for what we wanted Spark to do. This functionality allows Spark to perform the most efficient set of operations to get the desired result. The code example is the same as the previous slide, but with the added count() method call. This classifies as an action in Spark and will process all the transformation operations.

#### 6. Let's practice!

These concepts can be a little tricky to grasp without some examples. Let's practice these ideas in the coming exercises.



### Immutability review

You’ve just seen that immutability and lazy processing are fundamental concepts in the way Spark handles data. But why would Spark use immutable data frames to begin with?

To efficiently handle data throughout the cluster.



### Using lazy processing



```python
# Load the CSV file
aa_dfw_df = spark.read.format('csv').options(Header=True).load('AA_DFW_2018.csv.gz')

# Add the airport column using the F.lower() method
aa_dfw_df = aa_dfw_df.withColumn('airport', F.lower(aa_dfw_df['Destination Airport']))

# Drop the Destination Airport column
aa_dfw_df = aa_dfw_df.drop(aa_dfw_df['Destination Airport'])

# Show the DataFrame
aa_dfw_df.show()
```



### Understanding Parquet

#### 1. Understanding Parquet

Welcome back! As we've seen, Spark can read in text and CSV files. While this gives us access to many data sources, it's not always the most convenient format to work with. Let's take a look at a few problems with CSV files.

#### 2. Difficulties with CSV files

Some common issues with CSV files include: The schema is not defined: there are no data types included, nor column names (beyond a header row). Using content containing a comma (or another delimiter) requires escaping. Using the escape character within content requires even further escaping. The available encoding formats are limited depending on the language used.

#### 3. Spark and CSV files

In addition to the issues with CSV files in general, Spark has some specific problems processing CSV data. CSV files are quite slow to import and parse. The files cannot be shared between workers during the import process. If no schema is defined, all data must be read before a schema can be inferred. Spark has feature known as predicate pushdown. Basically, this is the idea of ordering tasks to do the least amount of work. Filtering data prior to processing is one of the primary optimizations of predicate pushdown. This drastically reduces the amount of information that must be processed in large data sets. Unfortunately, you cannot filter the CSV data via predicate pushdown. Finally, Spark processes are often multi-step and may utilize an intermediate file representation. These representations allow data to be used later without regenerating the data from source. Using CSV would instead require a significant amount of extra work defining schemas, encoding formats, etc.

#### 4. The Parquet Format

**Parquet is a compressed columnar data format developed for use in any Hadoop based system.** This includes Spark, Hadoop, Apache Impala, and so forth. The Parquet format is structured with data accessible in chunks, allowing efficient read / write operations without processing the entire file. This structured format supports Spark's **predicate pushdown** functionality, providing significant performance improvement. Finally, **Parquet files automatically include schema information and handle data encoding.** This is perfect for intermediary or on-disk representation of processed data. Note that Parquet files are a binary file format and can only be used with the proper tools. This is in contrast to CSV files which can be edited with any text editor.

#### 5. Working with Parquet

Interacting with Parquet files is very straightforward. To read a parquet file into a Data Frame, you have two options. The first is using the `spark.read.format` method we've seen previously. The Data Frame, `df=spark.read.format('parquet').load('filename.parquet')` The second option is the shortcut version: The Data Frame, `df=spark.read.parquet('filename.parquet') `Typically, the shortcut version is the easiest to use but you can use them interchangeably. Writing parquet files is similar, using either: `df.write.format('parquet').save('filename.parquet') `or `df.write.parquet('filename.parquet')` The long-form versions of each permit extra option flags, such as when overwriting an existing parquet file.

#### 6. Parquet and SQL

Parquet files have various uses within Spark. We've discussed using them as an intermediate data format, but they also are perfect for performing SQL operations. To perform a SQL query against a Parquet file, we first need to create a Data Frame via the `spark.read.parquet method.` Once we have the Data Frame, we can use the `createOrReplaceTempView() `method to add an alias of the Parquet data as a SQL table. Finally, we run our query using normal SQL syntax and the `spark.sql` method. In this case, we're looking for all flights with a duration under 100 minutes. Because we're using Parquet as the backing store, we get all the performance benefits we've discussed previously (primarily defined schemas and the available use of predicate pushdown).

#### 7. Let's Practice!

You've seen a bit about what a Parquet file is and why we'd want to use them. Now, let's practice working with Parquet files.



### Saving a DataFrame in Parquet format

The `spark` object and the `df1` and `df2` DataFrames have been setup for you.



```python
# View the row count of df1 and df2
print("df1 Count: %d" % df1.count())
print("df2 Count: %d" % df2.count())

# Combine the DataFrames into one
df3 = df1.union(df2)

# Save the df3 DataFrame in Parquet format
df3.write.parquet('AA_DFW_ALL.parquet', mode='overwrite')

# Read the Parquet file into a new DataFrame and run a count
print(spark.read.parquet('AA_DFW_ALL.parquet').count())
```



### SQL and Parquet

The `spark` object and the `AA_DFW_ALL.parquet` file are available for you automatically.

```python
# Read the Parquet file into flights_df
flights_df = spark.read.parquet('AA_DFW_ALL.parquet')

# Register the temp table
flights_df.createOrReplaceTempView('flights')

# Run a SQL query of the average flight duration
avg_duration = spark.sql('SELECT avg(flight_duration) from flights').collect()[0]
print('The average flight time is: %d' % avg_duration)
```



## Manipulating DataFrames in the real world

A look at various techniques to modify the contents of DataFrames in Spark.

### DataFrame column operations

**Got It!**

#### 1. DataFrame column operations

Welcome back! In the first chapter, we've spent some time discussing the basics of Spark data and file handling. Let's now take a look at how to use Spark column operations to clean data.

#### 2. DataFrame refresher

Before we discuss manipulating DataFrames in depth, let's talk about some of their features. DataFrames are made up of rows & columns and are generally analogous to a database table. DataFrames are immutable: any change to the structure or content of the data creates a new DataFrame. DataFrames are modified through the use of transformations. An example is The .filter() command to only return rows where the name starts with the letter 'M'. Another operation is .select(), in this case returning only the name and position fields.

#### 3. Common DataFrame transformations

There are many different transformations for use on a DataFrame. They vary depending on what you'd like to do. Some common transformations include: The .filter() clause, which includes only rows that satisfy the requirements defined in the argument. This is analogous to the WHERE clause in SQL. Spark includes a .where() alias which you can use in place of .filter() if desired. This call returns only rows where the vote occurred after 1/1/2019. Another common option is the .select() method which returns the columns requested from the DataFrame. The .withColumn() method creates a new column in the DataFrame. The first argument is the name of the column, and the second is the command(s) to create it. In this case, we create a column called 'year' with just the year information. We also can use the .drop() method to remove a column from a DataFrame.

#### 4. Filtering data

Among the most common operations used when cleaning a DataFrame, filtering lets us use only the data matching our desired result. We can use .filter() for many tasks, such as: Removing null values. Removing odd entries, anything that doesn't fit our desired format. We can also split a DataFrame containing combined data (such as a syslog file). As mentioned previously, use the .filter() method to return only rows that meet the specified criteria. The .contains() function takes a string argument that the column must have to return true. You can negate these results using the tile (~) character.

#### 5. Column string transformations

Some of the most common operations used in data cleaning are modifying and converting strings. You will typically apply these to each column as a transformation. Many of these functions are in the pyspark.sql.functions library. For brevity, we'll import it as the alias 'F'. We use the .withColumn() function to create a new column called "upper" using pyspark.sql.functions.upper() on the name column. The "upper" column will contain uppercase versions of all names. We can create intermediary columns that are only for processing. This is useful to clarify complex transformations requiring multiple steps. In this instance, we call the .split() function with the name of the column and the space character to split on. This returns a list of words in a column called splits. A very common operation is converting string data to a different type, such as converting a string column to an integer. We use the .cast() function to perform the conversion to an IntegerType().

#### 6. ArrayType() column functions

While performing data cleaning with Spark, you may need to interact with ArrayType() columns. These are analogous to lists in normal python environments. One function we will use is .size(), which returns the number of items present in the specified ArrayType() argument. Another commonly used function for ArrayTypes is .getItem(). It takes an index argument and returns the item present at that index in the list column. Spark has many more transformations and utility functions available. When using Spark in production, make sure to reference the documentation for available options.

#### 7. Let's practice!

We've discussed some of the common operations used on Spark DataFrame columns. Let's practice some of these now.



### Filtering column content with Python

This is often one of the first steps in data cleaning - removing anything that is obviously outside the format. For this dataset, make sure to look at the original data and see what looks out of place for the `VOTER_NAME` column.

The `pyspark.sql.functions` library is already imported under the alias `F`.

```python
# Show the distinct VOTER_NAME entries
voter_df.select('VOTER_NAME').distinct().show(40, truncate=False)

# Filter voter_df where the VOTER_NAME is 1-20 characters in length
voter_df = voter_df.filter('length(VOTER_NAME) > 0 and length(VOTER_NAME) < 20')

# Filter out voter_df where the VOTER_NAME contains an underscore
voter_df = voter_df.filter(~ F.col('VOTER_NAME').contains('_'))

# Show the distinct VOTER_NAME entries again
voter_df.select('VOTER_NAME').distinct().show(40, truncate=False)
```





### Modifying DataFrame columns

Previously, you filtered out any rows that didn't conform to something generally resembling a name. Now based on your earlier work, your manager has asked you to create two new columns - `first_name` and `last_name`. She asks you to split the `VOTER_NAME` column into words on any space character. You'll treat the last word as the `last_name`, and all other words as the `first_name`. You'll be using some new functions in this exercise including `.split()`, `.size()`, and `.getItem()`. The `.getItem(index)` takes an integer value to return the appropriately numbered item in the column. The functions `.split()` and `.size()` are in the `pyspark.sql.functions` library.

The filtered voter DataFrame from your previous exercise is available as `voter_df`. The `pyspark.sql.functions` library is available under the alias `F`.



```python
# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn('splits', F.split(voter_df.VOTER_NAME, '\s+'))

# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn('first_name', voter_df.splits.getItem(0))

# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn('last_name', voter_df.splits.getItem(F.size('splits') - 1))

# Drop the splits column
voter_df = voter_df.drop('splits')

# Show the voter_df DataFrame
voter_df.show()
```





### Conditional DataFrame column operations

#### 1. Conditional DataFrame column operations

We've looked at some of the power available when using Spark's functions to filter and modify our Data Frames. Let's spend some time with some more advanced options.

#### 2. Conditional clauses

The DataFrame transformations we've covered thus far are blanket transformations, meaning they're applied regardless of the data. Often you'll want to conditionally change some aspect of the contents. Spark provides some built in conditional clauses which act similar to an if / then / else statement in a traditional programming environment. While it is possible to perform a traditional if / then / else style statement in Spark, it can lead to serious performance degradation as each row of a DataFrame would be evaluated independently. Using the optimized, built-in conditionals alleviates this. There are two components to the conditional clauses: .when(), and the optional .otherwise(). Let's look at them in more depth.

#### 3. Conditional example

The .when() clause is a method available from the pyspark.sql.functions library that is looking for two components: the if condition, and what to do if it evaluates to true. This is best seen from an example. Consider a DataFrame with the Name and Age columns. We can actually add an extra argument to our .select() method using the .when() clause. We select df.Name and df.Age as usual. For the third argument, we'll define a when conditional. If the Age column is 18 or up, we'll add the string "Adult". If the clause doesn't match, nothing is returned. Note that our returned DataFrame contains an unnamed column we didn't define using .withColumn(). The .select() function can create columns dynamically based on the arguments provided. Let's look at some more examples.

#### 4. Another example

You can chain multiple when statements together, similar to an if / else if structure. In this case, we define two .when() clauses and return Adult or Minor based on the Age column. You can chain as many when clauses together as required.

#### 5. Otherwise

In addition to .when() is the otherwise() clause. .otherwise() is analogous to the else statement. It takes a single argument, which is what to return, in case the when clause or clauses do not evaluate as True. In this example, we return "Adult" when the Age column is 18 or higher. Otherwise, we return "Minor". The resulting DataFrame is the same, but the method is different. While you can have multiple .when() statements chained together, you can only have a single .otherwise() per .when() chain.

#### 6. Let's practice!

Let's try a couple examples of using .when() and .otherwise() to modify some DataFrames!





### when() example



```python
# Add a column to voter_df for any voter with the title **Councilmember**
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == 'Councilmember', F.rand()))

# Show some of the DataFrame rows, noting whether the when clause worked
voter_df.show()
```

### When / Otherwise

The `voter_df` Data Frame is defined and available to you. The `pyspark.sql.functions` library is available as `F.` You can use `F.rand()` to generate the random value.

```python
# Add a column to voter_df for a voter based on their position
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == 'Councilmember', F.rand())
                               .when(voter_df.TITLE == 'Mayor', 2)
                               .otherwise(0))

# Show some of the DataFrame rows
voter_df.show()

# Use the .filter() clause with random_val
voter_df.filter(voter_df.random_val == 0).show()
```



### User defined functions

#### 1. User defined functions

We've looked at the built-in functions in Spark and have had great results using these. But let's consider what you would do if you needed to apply some custom logic to your data cleaning processes.

#### 2. Defined...

A user defined function, or UDF, is a Python method that the user writes to perform a specific bit of logic. Once written, the method is called via the `pyspark.sql.functions.udf()` method. The result is stored as a variable and can be called as a normal Spark function. Let's look at a couple examples.

#### 3. Reverse string UDF

Here is a fairly trivial example to illustrate how a UDF is defined. First, we define a python function. We'll call our function, `reverseString()`, with an argument called `mystr`. We'll use some python shorthand to reverse the string and return it. Don't worry about understanding how the return statement works, only that it will reverse the lettering of whatever is fed into it (ie, "help" becomes "pleh"). The next step is to wrap the function and store it in a variable for later use. We'll use the `pyspark.sql.functions.udf()` method. It takes two arguments - the name of the method you just defined, and the Spark data type it will return. This can be any of the options in `pyspark.sql.types`, and can even be a more complex type, including a fully defined schema object. Most often, you'll return either a simple object type, or perhaps an `ArrayType`. We'll call `udf` with our new method name, and use the `StringType()`, then store this as `udfReverseString()`. Finally, we use our new UDF to add a column to the user_df DataFrame within the .`withColumn()` method. Note that we pass the column we're interested in as the argument to `udfReverseString()`. The udf function is called for each row of the Data Frame. Under the hood, the udf function takes the value stored for the specified column (per row) and passes it to the python method. The result is fed back to the resulting DataFrame.

#### 4. Argument-less example

Another quick example is using a function that does not require an argument. We're defining our `sortingCap()` function to return one of the letters 'G', 'H', 'R', or 'S' at random. We still create our udf wrapped function, and define the return type as `StringType()`. The primary difference is calling the function, this time without passing in an argument as it is not required.

#### 5. Let's practice!

As always, the best way to learn is practice - let's create some user defined functions!







## 3. Improving Performance

Improve data cleaning tasks by increasing performance or reducing resource requirements.

### Caching

#### 1. Caching

Now that we've done some data cleaning tasks using Spark, let's look at how to improve the performance of running those tasks using caching.

#### 2. What is caching?

Caching in Spark refers to storing the results of a DataFrame **in memory or on disk of the processing nodes in a cluster.** Caching improves the speed for subsequent transformations or actions as the data likely no longer needs to be retrieved from the original data source. Using caching reduces the resource utilization of the cluster - there is less need to access the storage, networking, and CPU of the Spark nodes as the data is likely already present.

#### 3. Disadvantages of caching

There are a few disadvantages of caching you should be aware of. **Very large data sets may not fit in the memory reserved for cached DataFrames.** Depending on the later transformations requested, the cache may not do anything to help performance. If a data set does not stay cached in memory, it may be persisted to disk. **Depending on the disk configuration of a Spark cluster, this may not be a large performance improvement.** If you're reading from a local network resource and have slow local disk I/O, it may be better to avoid caching the objects. Finally, **the lifetime of a cached object is not guaranteed**. Spark handles regenerating DataFrames for you automatically, but this can cause delays in processing.

#### 4. Caching tips

Caching is incredibly useful, but only if you plan to use the DataFrame again. If you only need it for a single task, it's not worth caching. The best way to gauge performance with caching is to test various configurations. Try caching your DataFrames at various points in the processing cycle and check if it improves your processing time. Try to cache in memory or fast NVMe / SSD storage. While still slower than main memory modern SSD based storage is drastically faster than spinning disk. Local spinning hard drives can still be useful if you are processing large DataFrames that require a lot of steps to generate, or must be accessed over the Internet. Testing this is crucial. If normal caching doesn't seem to work, try creating intermediate Parquet representations like we did in Chapter 1. These can provide a checkpoint in case a job fails mid-task and can still be used with caching to further improve performance. Finally, you can manually stop caching a DataFrame when you're finished with it. This frees up cache resources for other DataFrames.

#### 5. Implementing caching

Implementing caching in Spark is simple. The primary way is to call the function `.cache()` on a DataFrame object prior to a given Action. It requires no arguments. One example is creating a DataFrame from some original CSV data. Prior to running a `.count()` on the data, we call `.cache()` to tell Spark to store it in cache. Another option is to call .cache() separately. Here we create an ID in one transformation. We then call `.cache()` on the DataFrame. When we call the `.show()` action, the voter_df DataFrame will be cached. If you're following closely, this means that `.cache()` is a Spark transformation - nothing is actually cached until an action is called.

```python
voter_df = spark.read.csv('voter_data.txt.gz')
voter_df.cache().count()
                    
voter_df = voter_df.withColumn('ID', monotonically_increasing_id())
voter_df = voter_df.cache()
voter_df.show()
```

#### 6. More cache operations

A couple other options are available with caching in Spark. To check if a DataFrame is cached, use the .is_cached boolean property which returns True (as in this case) or False. To un-cache a DataFrame, we call `.unpersist()` with no arguments. This removes the object from the cache.

```python
# Check .is_cached to determine cache status
print(voter_df.is_cached)

# Call .unpersist() when finished with DataFrame
voter_df.unpersist()
```

#### 7. Let's Practice!

We've discussed caching in depth - let's practice how to use it!



### Caching a DataFrame

The DataFrame `departures_df` is defined, but no actions have been performed.

```python
start_time = time.time()

# Add caching to the unique rows in departures_df
departures_df = departures_df.distinct().cache()

# Count the unique rows in departures_df, noting how long the operation takes
print("Counting %d rows took %f seconds" % (departures_df.count(), time.time() - start_time))

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print("Counting %d rows again took %f seconds" % (departures_df.count(), time.time() - start_time))
```







### Improve import performance

#### 1. Improve import performance

We've discussed the benefits of caching when working with Spark DataFrames. Let's look at how to improve the speed when getting data into a DataFrame.

#### 2. Spark clusters

Spark clusters consist of **two types of processes** - one **driver process** and as many **worker processes** as required. The driver handles task assignments and consolidation of the data results from the workers. The workers typically handle the actual transformation / action tasks of a Spark job. Once assigned tasks, they operate fairly independently and report results back to the driver. It is possible to have a single node Spark cluster (this is what we're using for this course) but you'll rarely see this in a production environment. There are different ways to run Spark clusters - the method used depends on your specific environment.

#### 3. Import performance

When importing data to Spark DataFrames, it's important to understand how the cluster implements the job. The process varies depending on the type of task, but it's safe to assume that the more import objects available, the better the cluster can divvy up the job. This may not matter on a single node cluster, but with a larger cluster each worker can take part in the import process. In clearer terms, one large file will perform considerably worse than many smaller ones. Depending on the configuration of your cluster, you may not be able to process larger files, but could easily handle the same amount of data split between smaller files. Note you can define a single import statement, even if there are multiple files. You can use any form of standard wildcard symbol when defining the import filename. While less important, if objects are about the same size, the cluster will perform better than having a mix of very large and very small objects.

#### 4. Schemas

If you remember from chapter one, we discussed the importance of Spark schemas. Well-defined schemas in Spark drastically improve import performance. Without a schema defined, import tasks require reading the data multiple times to infer structure. This is very slow when you have a lot of data. Spark may not define the objects in the data the same as you would. Spark schemas also provide validation on import. This can save steps with data cleaning jobs and improve the overall processing time.

#### 5. How to split objects

There are various effective ways to split an object (files mostly) into more smaller objects. The first is to use built-in **OS utilities** such as split, cut, or awk. An example using split uses the -l argument with the number of lines to have per file (10000 in this case). The -d argument tells split to use numeric suffixes. The last two arguments are the name of the file to be split and the prefix to be used. Assuming 'largefile' has 10M records, we would have files named chunk-0000 through chunk-9999. Another method is to **use python (or any other language) to split the objects up as we see fit**. Sometimes you may not have the tools available to split a large file. If you're going to be working with a DataFrame often, a simple method is **to read in the single file then write it back out as parquet**. We've done this in previous examples and it works well for later analysis even if the initial import is slow. It's important to note that if you're hitting limitations due to cluster sizing, try to do as little processing as possible before writing to parquet.

```python
# Write out to Parquet
df_csv = spark.read.csv('singlelargefile.csv')
df_csv.write.parquet('data.parquet')
df = spark.read.parquet('data.parquet')
```

#### 6. Let's practice!

Let's practice some of the import tricks we've discussed now.



### File size optimization

Consider if you're given 2 large data files on a cluster with 10 nodes. Each file contains 10M rows of roughly the same size. While working with your data, the responsiveness is acceptable but the initial read from the files takes a considerable period of time. Note that you are the only one who will use the data and it changes for each run.

Which of the following is the best option to improve performance?

Split the 2 files into 50 files of 400K rows each.



```python
# Import the full and split files into DataFrames
full_df = spark.read.csv('departures_full.txt.gz')
split_df = spark.read.csv('departures_0*.txt.gz')

# Print the count and run time for each DataFrame
start_time_a = time.time()
print("Total rows in full DataFrame:\t%d" % full_df.count())
print("Time to run: %f" % (time.time() - start_time_a))

start_time_b = time.time()
print("Total rows in split DataFrame:\t%d" % split_df.count())
print("Time to run: %f" % (time.time() - start_time_b))
```



### Cluster configurations



#### 1. Cluster sizing tips

We've just finished working with improving import performance in Spark. Let's take a look at cluster configurations.

#### 2. Configuration options

Spark has many available configuration settings controlling all aspects of the installation. These configurations can be modified to best match the specific needs for the cluster. The configurations are available in the configuration files, via the Spark web interface, and via the run-time code. Our test cluster is only accessible via command shell so we'll use the last option. To read a configuration setting, call spark.conf.get() with the name of the setting as the argument. To write a configuration setting, call spark.conf.set() with the name of the setting and the actual value as the function arguments.

#### 3. Cluster Types

Spark deployments can vary depending on the exact needs of the users. One large component of a deployment is the cluster management mechanism. Spark clusters can be: Single node clusters, deploying all components on a single system (physical / VM / container). Standalone clusters, with dedicated machines as the driver and workers. Managed clusters, meaning that the cluster components are handled by a third party cluster manager such as YARN, Mesos, or Kubernetes. In this course, we're using a single node cluster. Your production environment can vary wildly, but we'll discuss standalone clusters as the concepts flow across all management types.

#### 4. Driver

If you recall, there is one driver per Spark cluster. The driver is responsible for several things, including the following: Handling task assignment to the various nodes / processes in the cluster. The driver monitors the state of all processes and tasks and handles any task retries. The driver is also responsible for consolidating results from the other processes in the cluster. The driver handles any access to shared data and verifies each worker process has the necessary resources (code, data, etc). Given the importance of the driver, it is often worth increasing the specifications of the node compared to other systems. Doubling the memory compared to other nodes is recommended. This is useful for task monitoring and data consolidation tasks. As with all Spark systems, fast local storage is useful for running Spark in an ideal setup.

#### 5. Worker

A Spark worker handles running tasks assigned by the driver and communicates those results back to the driver. Ideally, the worker has a copy of all code, data, and access to the necessary resources required to complete a given task. If any of these are unavailable, the worker must pause to obtain the resources. When sizing a cluster, there are a few recommendations: Depending on the type of task, more worker nodes is often better than larger nodes. This can be especially obvious during import and export operations as there are more machines available to do the work. As with everything in Spark, test various configurations to find the correct balance for your workload. Assuming a cloud environment, 16 worker nodes may complete a job in an hour and cost $50 in resources. An 8 worker configuration might take 1.25 hrs but cost only half as much. Finally, workers can make use of fast local storage (SSD / NVMe) for caching, intermediate files, etc.

#### 6. Let's practice!

Now that we've discussed cluster sizing and configuration, let's practice working with these options!












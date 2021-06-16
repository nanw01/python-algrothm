

# Introduction to Data Engineering



## 1. What is data engineering?

Hi. My name is Vincent. I'm a Data and Software Engineer at DataCamp. If you've ever heard of data science, there's a good chance you've heard of data engineering as well. This course will help you take your first steps in the world of data engineering. All very exciting, so let's get started!.

## 2. What to expect

In the first chapter, we'll start off by introducing the concept of data engineering. In the second chapter, you'll learn more about the tools data engineers use. The third chapter is all about Extracting, Transforming and Loading data, or ETL. Finally, you'll get to have a peek behind the curtain in the case study on data engineering at DataCamp. But first, let's understand what data engineers do!

## 3. In comes the data engineer

Imagine this: you've been hired as a data scientist at a young startup. Tasked with predicting customer churn, you want to use a fancy machine learning technique that you have been honing for years. However, after a bit of digging around, you realize all of your data is scattered around many databases. Additionally, the data resides in tables that are optimized for applications to run, not for analyses. To make matters worse, some legacy code has caused a lot of the data to be corrupt. In your previous company, you never really had this problem, because all the data was available to you in an orderly fashion. You're getting desperate. In comes the data engineer to the rescue.

## 4. Data engineers: making your life easier

It is the data engineer's task to make your life as a data scientist easier. Do you need data that currently comes from several different sources? No problem, the data engineer extracts data from these sources and loads it into one single database ready to use. At the same time, they've optimized the database scheme so it becomes faster to query. They also removed corrupt data. In this sense, the data engineer is one of the most valuable people in a data-driven company that wants to scale up.

## 5. Definition of the job

Back in 2015, DataCamp published an infographic on precisely this: who does what in the data science industry. In this infographic, we described a data engineer as "an engineer that develops, constructs, tests, and maintains architectures such as databases and large-scale processing systems." A lot has changed since then, but the definition still holds up. The data engineer is focused on processing and handling massive amounts of data, and setting up clusters of machines to do the computing.

## 6. Data Engineer vs Data Scientist

Typically, the tasks of a data engineer consist of developing a scalable data architecture, streamlining data acquisition, setting up processes that bring data together from several sources and safeguarding data quality by cleaning up corrupt data. Typically, the data engineer also has a deep understanding of cloud technology. They generally are experienced using cloud service providers like AWS, Azure, or Google Cloud. Compare this with the tasks of a data scientist, who spend their time mining for patterns in data, applying statistical models on large datasets, building predictive models using machine learning, developing tools to monitor essential business processes, or cleaning data by removing statistical outliers. Data scientist typically have a deep understanding of the business itself.

![image-20210323163621102](https://i.loli.net/2021/03/24/MAOfPRgFEk25HTb.png)

## 7. Let's practice!

Let's see if you can recognize the qualities of a data engineer in the exercises.







# Tools of the data engineer

## 1. Tools of the data engineer

Hello again. Great job on the exercises! You should now have a good understanding of what it means to be a data engineer. The data engineer moves data from several sources, processes or cleans it and finally loads it into an analytical database. They do this using several tools. This video acts as an overview to get a feeling for how data engineers fulfill their tasks using these tools. We'll spend some more time to go into the details in the second chapter.

## 2. Databases

First, data engineers are expert users of database systems. Roughly speaking, a database is a computer system that holds large amounts of data. You might have heard of SQL or NoSQL databases. If not, there are some excellent courses on DataCamp on these subjects. Often, applications rely on databases to provide certain functionality. For example, in an online store, a database holds product data like prices or amount in stock. On the other hand, other databases hold data specifically for analyses. You'll find out more about the difference in later chapters. For now, it's essential to understand that the data engineer's task begins and ends at databases.

## 3. Processing

Second, data engineers use tools that can help them quickly process data. Processing data might be necessary to clean or aggregate data or to join it together from different sources. Typically, huge amounts of data have to be processed. That is where parallel processing comes into play. Instead of processing the data on one computer, data engineers use clusters of machines to process the data. Often, these tools make an abstraction of the underlying architecture and have a simple API.

## 4. Processing: an example

For example, have a look at this code. It looks a lot like simple pandas filter or count operations. However, behind the curtains, a cluster of computers could be performing these operations using the PySpark framework. We'll get into the details of different parallel processing frameworks later, but a good data engineer understands these abstractions and knows their limitations.

## 5. Scheduling

Third, scheduling tools help to make sure data moves from one place to another at the correct time, with a specific interval. Data engineers make sure these jobs run in a timely fashion and that they run in the right order. Sometimes processing jobs need to run in a particular order to function correctly. For example, tables from two databases might need to be joined together after they are both cleaned. In the following diagram, the JoinProductOrder job needs to run after CleanProduct and CleanOrder ran.

## 6. Existing tools

Luckily all of these tools are so common that there is a lot of choice in deciding which ones to use. In this slide, I'll present a few examples of each kind of tool. Please keep in mind this list is not exhaustive, and that some companies might choose to build their own tools in-house. Two examples of databases are MySQL or PostgreSQL. An example processing tool is Spark or Hive. Finally, for scheduling, we can use Apache Airflow, Oozie, or we can use the simple bash tool: cron.

## 7. A data pipeline

To sum everything up, you can think of the data engineering pipeline through this diagram. It extracts all data through connections with several databases, transforms it using a cluster computing framework like Spark, and loads it into an analytical database. Also, everything is scheduled to run in a specific order through a scheduling framework like Airflow. A small side note here is that the sources can be external APIs or other file formats too. We'll see this in the exercises.

## 8. Let's practice!

Enough talking, let's do some exercises!



# Cloud providers

## 1. Cloud providers

Hello again. Excellent work on the exercises! In this last video of this chapter, we're going to talk about cloud computing. You've probably already heard people use this term before. Data engineers are heavy users of the cloud. In this video, we'll explain why.

## 2. Data processing in the cloud

Let's take data processing as an example. You've seen in the previous video that data processing often runs on clusters of machines. In the past, companies that relied on data processing owned their own data center. You can imagine racks of servers, ready to be used. The electrical bill and maintenance were also at the company's cost. Moreover, companies needed to be able to provide enough processing power for peak moments. That also meant that at quieter times, much of the processing power remained unused. It's this waste of resources that made cloud computing so appealing. In the cloud, you use the resources you need, at the time you need them. You can see that once these cloud services came to be, many companies moved to the cloud as a way of cost optimization.

## 3. Data storage in the cloud

Apart from the costs of maintaining data centers, another reason for using cloud computing is database reliability. If you run a data-critical company, you have to prepare for the worst. Don't ask yourself the question "will disaster strike?" but rather ask yourself "when will disaster strike?" For example, a fire can break out in your data center. To be safe, you need to replicate your data at a different geographical location. That brings along a bunch of logistical problems of its own. Out of these needs, companies specializing in these kinds of issues were born. We call these companies "cloud service providers" now.

## 4. The big three: AWS, Azure and Google

In this slide, we'll talk about three big players in the cloud provider market. First and foremost, there's Amazon Web Services or AWS. Think about the last few websites you visited. Chances are AWS hosts at least a few of them. Back in 2017, AWS had an outage, it reportedly 'broke' the internet. That's how big AWS is. While AWS took up 32% of the market share in 2018, Microsoft Azure is the second big player and took 17% of the market. The third big player, is Google Cloud, and held 10% of the market in 2018. So we talked about the big players. However, what do they provide? We'll discuss three types of services these companies offer: Storage, Computation, and Databases.

## 5. Storage

First, storage services allow you to upload files of all types to the cloud. In an online store for example, you could upload your product images to a storage service. Storage services are typically very cheap since they don't provide much functionality other than storing the files reliably. **AWS hosts S3 as a storage service. Azure has Blob Storage, and Google has Cloud Storage.**

## 6. Computation

Second, computation services allow you to perform computations on the cloud. Usually, you can start up a virtual machine and use it as you wish. It's often used to host web servers, for example. Computation services are usually flexible, and you can start or stop virtual machines as needed. **AWS has EC2 as a computation service, Azure has Virtual Machines, and Google has Compute Engine.**

## 7. Databases

Last but not least, cloud providers host databases. We already talked about databases in the previous video, so you know what they are. For SQL databases, **AWS has RDS. Azure has SQL Database, and Google has Cloud SQL.**

## 8. Let's practice!

That's it for this video, good luck with the exercises.



![image-20210323200218073](https://i.loli.net/2021/03/24/JEtuocBTMeDkwG5.png)







# Data engineering toolbox

# Databases

## 1. Databases

Hi and welcome back! In the last chapter, we learned what a data engineer is. In this chapter, you'll learn everything about the tools of a data engineer. Let's start with databases.

## 2. What are databases?

Databases are an essential tool for the Data Engineer. They can be used to store information. Before zooming in to the kinds of databases, let's get some definitions out of the way. According to Merriam-Webster, **a database is "a usually large collection of data organized especially for rapid search and retrieval."** There are few pieces of vital information in this definition. First, the database **holds data**. Second, databases **organize data**. We'll see later that there are differences in the level of organization between database types. Lastly, databases help us **quickly retrieve or search for data**. The database management system or DBMS is usually in charge of this.

## 3. Databases and file storage

The main difference between databases and simple storage systems like file systems is **the level of organization** and the fact that the database management systems abstract away a lot of **complicated data operations** like search, replication and much more. File systems host less such functionality.

## 4. Structured and unstructured data

In the universe of databases, there's a big difference in the level of organization. To understand these differences, we have to make a distinction between structured and unstructured data. On one hand, **structured data is coherent to a well-defined structure.** **Database schemas** usually define such structure. An example of structured data is **tabular data in a relational database.** **Unstructured data**, on the other hand, is **schemaless**. It looks a lot more like files. Unstructured data could be something **like photographs or videos.** Structured and unstructured data define outer boundaries, and there is a whole lot of semi-structured data in between. An example of **semi-structured data is JSON data**.

## 5. SQL and NoSQL

Another distinction we can make is the one between SQL and NoSQL. Generally speaking, in SQL databases, tables form the data. The database schema defines relations between these tables. We call SQL databases relational. For example, we could create one table for customers and another for orders. The database schema defines the relationships and properties. Typical SQL databases are MySQL and PostgreSQL. On the other hand, NoSQL databases are called non-relational. NoSQL is often associated with unstructured, schemaless data. That's a misconception, as there are several types of NoSQL databases and they are not all unstructured. Two highly used NoSQL database types are key-value stores like Redis or document databases like MongoDB. In key-value stores, the values are simple. Typical use cases are caching or distributed configuration. Values in a document database are structured or semi-structured objects, for example, a JSON object.

![image-20210323200854828](https://i.loli.net/2021/03/24/Uou7JLmkpNHOTVr.png)

## 6. SQL: The database schema

For the remainder of this video, let's focus on database schemas. A schema describes the structure and relations of a database. In this slide, you can see a database schema on the left-hand side. It represents the relations shown in the diagram to the right. We see a table called Customer and one called Order. The column called customer_id connects orders with customers. We call this kind of column a foreign key, as it refers to another table. The SQL statements on the left create the tables of the schema. As you've seen in courses on SQL, you can leverage these foreign keys by joining tables using the JOIN statement.

## 7. SQL: Star schema

In data warehousing, a schema you'll see often is the star schema. A lot of analytical databases like Redshift have optimizations for these kinds of schemas. According to Wikipedia, "the star schema consists of one or more fact tables referencing any number of dimension tables." Fact tables contain records that represent things that happened in the world, like orders. Dimension tables hold information on the world itself, like customer names or product prices.

1. 1 Wikipedia: https://en.wikipedia.org/wiki/Star_schema

## 8. Let's practice!

Now that you know about databases, let's practice in the exercises!







**star schema**

![image-20210323201754505](https://i.loli.net/2021/03/24/W8Heh9niEVusXCS.png)

**B**





# What is parallel computing

**Got It!**

## 1. What is parallel computing

Hi again! Now that you've learned everything about databases let's talk about parallel computing. In data engineering, you often have to pull in data from several sources and join them together, clean them, or aggregate them. In this video, we'll see how this is possible for massive amounts of data.

## 2. Idea behind parallel computing

Before we go into the different kinds of tools that exist in the data engineering ecosystem, it's crucial to understand the concept of parallel computing. Parallel computing forms the basis of almost all modern data processing tools. However, why has it become so important in the world of big data? The main reason is memory and processing power, but mostly memory. When big data processing tools perform a processing task, they split it up into several smaller subtasks. The processing tools then distribute these subtasks over several computers. These are usually commodity computers, which means they are widely available and relatively inexpensive. Individually, all of the computers would take a long time to process the complete task. However, since all the computers work in parallel on smaller subtasks, the task in its whole is done faster.

![image-20210323202111048](https://i.loli.net/2021/03/24/nCQ6AfJ2yYp4eK8.png)

## 3. The tailor shop

Let's look at an analogy. Let's say you're running a tailor shop and need to get a batch of 100 shirts finished. Your very best tailor finishes a shirt in 20 minutes. Other tailors typically take 1 hour per shirt. If just one tailor can work at a time, it's obvious you'd have to choose the quickest tailor to finish the job. However, if you can split the batch in 25 shirts each, having 4 mediocre tailors working in parallel is faster. A similar thing happens for big data processing tasks.

## 4. Benefits of parallel computing

As you'd expect, the obvious benefit of having multiple processing units is the extra **processing power** itself. However, there is another, and potentially more impactful benefit of parallel computing for big data. **Instead of needing to load all of the data in one computer's memory, you can partition the data and load the subsets into memory of different computers.** That means the memory footprint per computer is relatively small, and the data can fit in the memory closest to the processor, the RAM.

## 5. Risks of parallel computing

Before you start rewriting all your code to use parallel computing, keep in mind that this also comes at its **cost**. **Splitting a task into subtask and merging the results of the subtasks back into one final result requires some communication between processes.** This communication overhead **can become a bottleneck** if the processing requirements are not substantial, or if you have too little processing units. In other words, if you have 2 processing units, a task that takes a few hundred milliseconds might not be worth splitting up. Additionally, due to the overhead, the speed does not increase linearly. This effect is also called **parallel slowdown.**

## 6. An example

Let's look into a more practical example. We're starting with a dataset of all Olympic events from 1896 until 2016. From this dataset, you want to get an average age of participants for each year. For this example, let's say you have four processing units at your disposal. You decide to distribute the load over all of your processing units. To do so, you need to split the task into smaller subtasks. In this example, the average age calculation for each group of years is as a subtask. You can achieve that through 'groupby.' Then, you distribute all of these subtasks over the four processing units. This example illustrates roughly how the first distributed algorithms like Hadoop MapReduce work, the difference being the processing units are distributed over several machines.

## 7. multiprocessing.Pool

In code, there are several ways of implementing this. At a low level, we could use the `multiprocessing.Pool` API to distribute work over several cores on the same machine. Let's say we have a function `take_mean_age`, which accepts a tuple: the year of the group and the group itself as a DataFrame. `take_mean_age` returns a DataFrame with one observation and one column: the mean age of the group. The resulting DataFrame is indexed by year. We can then take this function, and map it over the groups generated by `.groupby()`, using the `.map()` method of `Pool`. By defining `4` as an argument to `Pool`, the mapping runs in 4 separate processes, and thus uses 4 cores. Finally, we can concatenate the results to form the resulting DataFrame.

## 8. dask

Several packages offer a layer of abstraction to avoid having to write such low-level code. For example, the `dask` framework offers a DataFrame object, which performs a groupby and apply using multiprocessing out of the box. You need to define the number of partitions, for example, `4`. `dask` divides the DataFrame into 4 parts, and performs `.mean()` within each part separately. Because `dask` uses lazy evaluation, you need to add `.compute()` to the end of the chain.

## 9. Let's practice!

That was the final example of this video. In the exercises, you'll use the packages yourself. Good luck!



# Why parallel computing?

You've seen the benefits of parallel computing. However, you've also seen it's not the silver bullet to fix all problems related to computing.

- Parallel computing can optimize the use of multiple processing units.

- Parallel computing can optimize the use of memory between several machines.

  



## From task to subtasks

```python
# Function to apply a function over multiple cores
@print_timing
def parallel_apply(apply_func, groups, nb_cores):
    with Pool(nb_cores) as p:
        results = p.map(apply_func, groups)
    return pd.concat(results)

# Parallel apply using 1 core
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 1)

# Parallel apply using 2 cores
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 2)

# Parallel apply using 4 cores
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 4)
```

## Using a DataFrame

```python
import dask.dataframe as dd

# Set the number of pratitions
athlete_events_dask = dd.from_pandas(athlete_events, npartitions = 4)

# Calculate the mean Age per Year
print(athlete_events_dask.groupby('Year').Age.mean().compute())
```





# Parallel computation frameworks

**Got It!**

## 1. Parallel computation frameworks

Hi again. Now that we've seen the basics of parallel computing, it's time to talk about specific parallel computing frameworks. We'll focus on parallel computing frameworks that are currently hot in the data engineering world.

## 2. Hadoop

Before we dive into state of the art big data processing tools, let's talk a bit about the foundations. If you've done some investigation into big data ecosystems, you've probably heard about Hadoop. Hadoop is a collection of open source projects, maintained by the Apache Software Foundation. Some of them are a bit outdated, but it's still relevant to talk about them. There are two Hadoop projects we want to focus on for this video: MapReduce and HDFS.

## 3. HDFS

**First, HDFS is a distributed file system.** It's similar to the file system you have on your computer, the only difference being the files reside on multiple different computers. HDFS has been essential in the big data world, and for parallel computing by extension. Nowadays, cloud-managed storage systems like Amazon S3 often replace HDFS.

## 4. MapReduce

Second, **MapReduce was one of the first popularized big-data processing paradigms.** It works similar to the example you saw in the previous video, where the program splits tasks into subtasks, distributing the workload and data between several processing units. For MapReduce, these processing units are several computers in the cluster. MapReduce had its flaws; one of it was that it was hard to write these MapReduce jobs. Many software programs popped up to address this problem, and one of them was Hive.

## 5. Hive

**Hive is a layer on top of the Hadoop ecosystem that makes data from several sources queryable in a structured way using Hive's SQL variant: Hive SQL.** Facebook initially developed Hive, but the Apache Software Foundation now maintains the project. Although MapReduce was initially responsible for running the Hive jobs, it now integrates with several other data processing tools.

## 6. Hive: an example

Let's look at an example: this Hive query selects the average age of the Olympians per Year they participated. As you'd expect, this query looks indistinguishable from a regular SQL query. However, behind the curtains, this query is transformed into a job that can operate on a cluster of computers.

## 7. Spark

The other **parallel computation framework** we'll introduce is called Spark. **Spark distributes data processing tasks between clusters of computers.** While MapReduce-based systems tend to need expensive disk writes between jobs, Spark tries to keep as much processing as possible in memory. In that sense, Spark was also an answer to the limitations of MapReduce. The disk writes of MapReduce were especially limiting in interactive exploratory data analysis, where each step builds on top of a previous step. Spark originates from the University of California, where it was developed at the Berkeley's AMPLab. Currently, the Apache Software Foundation maintains the project.

## 8. Resilient distributed datasets (RDD)

**Spark's architecture relies on something called resilient distributed datasets, or RDDs.** Without diving into technicalities, this is a data structure that maintains data which is distributed between multiple nodes. Unlike DataFrames, RDDs don't have named columns. From a conceptual perspective, you **can think of RDDs as lists of tuples.** We can do two types of operations on these data structures: **transformations**, **like map or filter**, and **actions**, **like count or first**. Transformations result in transformed RDDs, while actions result in a single result.

## 9. PySpark

When working with Spark, people typically use a programming language interface like PySpark. PySpark is the Python interface to spark. There are interfaces to Spark in other languages, like R or Scala, as well. PySpark hosts a DataFrame abstraction, which means that you can do operations very similar to pandas DataFrames. PySpark and Spark take care of all the complex parallel computing operations.

## 10. PySpark: an example

Have a look at the following PySpark example. Similar to the Hive Query you saw before, it calculates the mean age of the Olympians, per Year of the Olympic event. Instead of using the SQL abstraction, like in the Hive Example, it uses the DataFrame abstraction.

## 11. Let's practice!

Let's try this in the exercises!

![image-20210323205003561](https://i.loli.net/2021/03/24/n9ZtVGvLJhEoImp.png)

## A PySpark groupby

```python
# Print the type of athlete_events_spark
print(type(athlete_events_spark))

# Print the schema of athlete_events_spark
print(athlete_events_spark.printSchema())

# Group by the Year, and find the mean Age
print(athlete_events_spark.groupBy('Year').mean('Age'))

# Group by the Year, and find the mean Age
print(athlete_events_spark.groupBy('Year').mean('Age').show())
```

#### Running PySpark files

In this exercise, you're going to run a PySpark file using `spark-submit`. This tool can help you submit your application to a spark cluster.

For the sake of this exercise, you're going to work with a local Spark instance running on 4 threads. The file you need to submit is in `/home/repl/spark-script.py`. Feel free to read the file:

```
cat /home/repl/spark-script.py
```

You can use `spark-submit` as follows:

```
spark-submit \
  --master local[4] \
  /home/repl/spark-script.py
```



# Workflow scheduling frameworks

## 1. Workflow scheduling frameworks

Hi, and welcome to the last video of the chapter. In this video, we'll introduce the last tool we'll talk about in this course. These are the **workflow scheduling frameworks**. We've seen how to pull data from existing databases. We've also seen parallel computing frameworks like Spark. However, something needs to put all of these jobs together. It's the task of the workflow scheduling framework to orchestrate these jobs.

## 2. An example pipeline

Let's take an example. You can write a Spark job that pulls data from a CSV file, filters out some corrupt records, and loads the data into a SQL database ready for analysis. However, let's say you need to do this every day as new data is coming in to the CSV file. One option is to run the job each day manually. Of course, that doesn't scale well: what about the weekends? There are simple tools that could solve this problem, like cron, the Linux tool. However, let's say you have one job for the CSV file and another job to pull in and clean the data from an API, and a third job that joins the data from the CSV and the API together. The third job depends on the first two jobs to finish. It quickly becomes apparent that we need a more holistic approach, and a simple tool like cron won't suffice.

## 3. DAGs

So, we ended up with dependencies between jobs. A great way to visualize these dependencies is through **Directed Acyclic Graphs**, or DAGs. A DAG is a set of nodes that are connected by directed edges. There are no cycles in the graph, which means that no path following the directed edges sees a specific node more than once. In the example on the slide, Job A needs to happen first, then Job B, which enables Job C and D and finally Job E. As you can see, it feels natural to represent this kind of workflow in a DAG. The jobs represented by the DAG can then run in a daily schedule, for example.

## 4. The tools for the job

So, we've talked about dependencies and scheduling DAGs, what tools are there to use for us? Well, first of all, some people use the **Linux tool, cron.** However, **most companies use a more full-fledged solution.** There's **Spotify's Luigi**, which allows for the definition of DAGs for complex pipelines. However, for the remainder of the video, we'll focus on **Apache Airflow**. Airflow is growing out to be the de-facto workflow scheduling framework.

## 5. Apache Airflow

Airbnb created Airflow as an internal tool for workflow management. They open-sourced Airflow in 2015, and it later joined the Apache Software Foundation in 2016. They built Airflow around the concept of DAGs. Using Python, developers can create and test these DAGs that build up complex pipelines.

## 6. Airflow: an example DAG

Let's look at an example from an e-commerce use-case in the DAG showed on the slide. The first job starts a Spark cluster. Once it's started, we can pull in customer and product data by running the ingest_customer_data and ingest_product_data jobs. Finally, we aggregate both tables using the enrich_customer_data job which runs after both ingest_customer_data and ingest_product_data complete.

## 7. Airflow: an example in code

In code, it would look something like this. First, we create a DAG using the `DAG` class. Afterward, we use an Operator to define each of the jobs. Several kinds of operators exist in Airflow. There are simple ones like BashOperator and PythonOperator that execute bash or Python code, respectively. Then there are ways to write your own operator, like the SparkJobOperator or StartClusterOperator in the example. Finally, we define the connections between these operators using `.set_downstream()`.

```python
# Create the DAG object
dag = DAG(dag_id="example_dag", ..., schedule_interval="0 * * * *")
# Define operations
start_cluster = StartClusterOperator(
    task_id="start_cluster", 
    dag=dag
)
ingest_customer_data = SparkJobOperator(
    task_id="ingest_customer_data", 
    dag=dag
)
ingest_product_data = SparkJobOperator(
    task_id="ingest_product_data", 
    dag=dag
)
enrich_customer_data = PythonOperator(
    task_id="enrich_customer_data",
    ...,
    dag = dag
)
# Set up dependency flow
start_cluster.set_downstream(ingest_customer_data)
ingest_customer_data.set_downstream(enrich_customer_data)
ingest_product_data.set_downstream(enrich_customer_data)
```

## 8. Let's practice!

Ok, let's see how you do in the exercises.

```python
# Create the DAG object
dag = DAG(dag_id="car_factory_simulation",
          default_args={"owner": "airflow","start_date": airflow.utils.dates.days_ago(2)},
          schedule_interval="0 * * * *")

# Task definitions
assemble_frame = BashOperator(task_id="assemble_frame", bash_command='echo "Assembling frame"', dag=dag)
place_tires = BashOperator(task_id="place_tires", bash_command='echo "Placing tires"', dag=dag)
assemble_body = BashOperator(task_id="assemble_body", bash_command='echo "Assembling body"', dag=dag)
apply_paint = BashOperator(task_id="apply_paint", bash_command='echo "Applying paint"', dag=dag)

# Complete the downstream flow
assemble_frame.set_downstream(place_tires)
assemble_frame.set_downstream(assemble_body)
assemble_body.set_downstream(apply_paint)
```



# Extract, Transform and Load (ETL)

# Extract

**Got It!**

## 1. Extract

You made it to the third chapter. Impressive work so far. This chapter covers a concept that we often refer to as ETL in data engineering. **ETL stands for Extract, Transform, and Load.** We'll have one lesson on each of these steps. In the final lesson, we'll set up an ETL process using a scheduler we saw in the previous chapter.

## 2. Extracting data: what does it mean?

This first lesson is about data extraction or the extract phase in ETL. Now, what do we mean by extracting data? Very roughly, this means extracting data from persistent storage, which is not suited for data processing, into memory. **Persistent storage** could be a file on Amazon S3, for example, or a SQL database. It's the necessary stage before we can start transforming the data. The sources to extract from vary.

## 3. Extract from text files

First of all, we can extract data from plain text files. These are files that are generally readable for people. They can be unstructured, like a chapter from a book like Moby Dick. Alternatively, these can be flat files, where each row is a record, and each column is an attribute of the records. In the latter, we represent data in a tabular format. **Typical examples of flat files are comma-, or tab-separated files: .csv or .tsv.** They use commas (,) or tabs respectively to separate columns.

## 4. JSON

Another widespread data format is called JSON, or JavaScript Object Notation. JSON files hold information in a semi-structured way. It consists of 4 atomic data types: number, string, boolean and null. There are also 2 composite data types: array and object. You could compare it to a dictionary in Python. JSON objects can be very nested, like in this example. There's a pretty good mapping from JSON objects to dictionaries in Python. There's a package in the standard library called `json`, which helps you parse JSON data. The function `json.loads` helps you with this. The reason JSON got popular is that in recent days, many web services use this data format to communicate data.

## 5. Data on the Web

At this point, it makes sense to do a crash course on the web. On the web, most communication happens to something called 'requests.' You can look at a request as a 'request for data.' A request gets a response. For example, if you browse Google in your web browser, your browser requests the content of the Google home page. Google servers respond with the data that makes up the page.

## 6. Data on the Web through APIs

However, some web servers don't serve web pages that need to be readable by humans. Some serve data in a JSON data format. We call these servers APIs or application programming interfaces. The popular social media tool, Twitter, hosts an API that provides us with information on tweets in JSON format. Using the Twitter API does not tell us anything about how Twitter stores its data; it merely provides us with a structured way of querying their data. Let's look at another example request to the Hackernews API and the resulting JSON response. As you can see, you can use the Python package, `requests` to request an API. We will use the `.get()` method and pass an URL. The resulting response object has a built-in helper method called `.json()` to parse the incoming JSON and transform it into a Python object.

```python
import requests
response = requests.get("https://hacker-news.firebaseio.com/v0/item/16222426.json")
print(response.json())
```

```json
{"by":"neis","descendants":0,"id":16222426,"score":17,"time":1516800333,"title":"Duolingo-Style Learning for Data Science: DataCamp for Mobile","type":"story","url":"https://medium.com/datacamp/duolingo-style-learning-for-data-science-datacamp-for-mobile-3861d1bc02df"}
```



## 7. Data in databases

Finally, we have to talk about databases. The most common way of data extraction is extraction from existing application databases. Most applications, like web services, need a database to back them up and **persist data**. At this point, it's essential to make a distinction between two main database types. Databases that applications like web services use, are typically optimized for having lots of **transactions**. A transaction typically **changes** or **inserts** rows, or records, in the database. For example, let's say we have a customer database. Each row, or record, represents data for one specific customer. **A transaction could add a customer to the database, or change their address.** These kinds of transactional databases are called **OLTP, or online transaction processing.** They are typically row-oriented, in which the system adds data per rows. In contrast, **databases optimized for analysis are called OLAP, or online analytical processing.** They are often column-oriented. We'll talk more about this later in this chapter.

![image-20210323212130198](https://i.loli.net/2021/03/24/IlByLtdaMYZNObz.png)

## 8. Extraction from databases

To extract data from a database in Python, you'll always need a connection string. The connection string or connection URI is a string that holds information on how to connect to a database. It typically contains the database type, for example, PostgreSQL, the username and password, the host and port and the database name. In Python, you'd use this connection URI with a package like `sqlalchemy` to create a database engine. We can pass this engine object to several packages that support it to interact with the database. The example shows the usage with `pandas`.

```python
# postgresql://[user[:password]@][host][:port]
    
import sqlalchemy
connection_uri = "postgresql://repl:password@localhost:5432/pagila"
db_engine = sqlalchemy.create_engine(connection_uri)

import pandas as pd
pd.read_sql("SELECT * FROM customer", db_engine)
```



## 9. Let's practice!

Now that know saw the extract phase, let's look at some exercises!



# Data sources

In the previous video you've learned about three ways of extracting data:

- Extract from text files, like `.txt` or `.csv`
- Extract from APIs of web services, like the Hacker News API
- Extract from a database, like a SQL application database for customer data

We also briefly touched upon row-oriented databases and OLTP.

Can you select the statement about these topics which is true**?

- OLTP means the system is optimized for transactions.
- Row-oriented databases and OLTP go hand-in-hand.



```python
import requests

# Fetch the Hackernews post
resp = requests.get("https://hacker-news.firebaseio.com/v0/item/16222426.json")

# Print the response parsed as JSON
print(resp.json())

# Assign the score of the test to post_score
post_score = resp.json()['score']
print(post_score)
```



```python
# Function to extract table to a pandas DataFrame
def extract_table_to_pandas(tablename, db_engine):
    query = "SELECT * FROM {}".format(tablename)
    return pd.read_sql(query, db_engine)

# Connect to the database using the connection URI
connection_uri = "postgresql://repl:password@localhost:5432/pagila" 
db_engine = sqlalchemy.create_engine(connection_uri)

# Extract the film table into a pandas DataFrame
extract_table_to_pandas('film', db_engine)

# Extract the customer table into a pandas DataFrame
extract_table_to_pandas('customer', db_engine)
```



# Transform

## 1. Transform

Great job on those exercises. In this video, we'll focus on the second stage of the ETL pipeline: transform.

## 2. Kind of transformations

We've already talked a bit about transformations in the previous chapter when we talked about parallel computing. We saw how transformations are typically done using parallel computing. However, we didn't talk about what kind of transformations a data engineer has to do. To illustrate this, have a look at the following sample record of a customer database from a DVD Store. It contains a customer id, an email, the state they live in and the date they created their account. Here's a non-exhaustive list of things the data engineer might have to perform on this data during the transform phase. First, there's the selection of specific attribute, for example, we could select the 'email' column only. Second, there is the translation of code values. For instance, 'New York' could be translated into 'NY'. Third, the transformation phase could validate the data. For example, if 'created_at' does not contain a date value, we could drop the record. For the last two transformations, splitting and joining, we'll go into some more detailed examples.

## 3. An example: split (Pandas)

In the first detailed example, the goal is to split up a single column into multiple columns. You might want to have the e-mail address attribute split into username and domain name. For this example, we'll use Pandas for the transformation. To achieve this, you can use the `str.split()` method on the `customer_df.email` Pandas Series. After the split, you can add two new columns: `username` and `domain`.

```python
customer_df 
# Pandas DataFrame with customer data
# Split email column into 2 columns on the '@' symbol
split_email = customer_df.email.str.split("@", expand=True)

# At this point, split_email will have 2 columns, a first
# one with everything before @, and a second one with
# everything after @

# Create 2 new columns using the resulting DataFrame.
customer_df = customer_df.assign(username=split_email[0],domain=split_email[1],)
```



## 4. Transforming in PySpark

Before moving on to the final illustrative example of joining, we need to take a quick detour. The last transformation example will be using PySpark. We could just as well have used pandas if the load is small. However, since we used PySpark, the extract phase needs to load the table into Spark. We can do this with `spark.read.jdbc`, where `spark` is a `SparkSession` object. JDBC is a piece of software that helps Spark connect to several relational databases. There are some differences between this connection URI and the one you saw in the previous video. First of all, it's prepended by 'jdbc:', to tell Spark to use JDBC. Second, we pass authorization information in the `properties` attribute instead of the URL. Finally, we pass the name of the table as a second argument. If you are interested in learning about Spark, DataCamp offers other courses that delve deeper into Spark.

```python
import pyspark.sql
spark = pyspark.sql.SparkSession.builder.getOrCreate()
spark.read.jdbc("jdbc:postgresql://localhost:5432/pagila","customer",properties={"user":"repl","password":"password"})
```



## 5. An example: join

Now, let's move on to the final example. Let's say that as a marketing effort, we allow users to use their mobile phone to rate the films they watched. An external firm has created the service, and you end up with a new database containing ratings from customers for specific films. For simplicity's sake, we'll assume that we use the same film and customer ids as in the store's database. A transformation phase could use a table from the store's database and the rating's database. For example, we could add the mean rating for each customer as an attribute to the customer table.

```python
customer_df # PySpark DataFrame with customer data
ratings_df # PySpark DataFrame with ratings data
# Groupby ratings
ratings_per_customer = ratings_df.groupBy("customer_id").mean("rating")
# Join on customer ID
customer_df.join(ratings_per_customer,customer_df.customer_id==ratings_per_customer.customer_id)
```



## 6. An example: join (PySpark)

Now, how to do this in code with PySpark? Let's say you have two DataFrames, customer and ratings. We want to figure out the mean rating for each customer and add it to the customer dataframe. First, we aggregate the ratings by grouping by customer ids using the .groupBy() method. To get the mean rating per customer, we chain the groupby method with the .mean() method. Afterward, we can join the aggregated table with the customer table. That gives us the customer table, extended with the mean rating for each customer. Note how we set the matching keys of the two data frames when joining the data frames.

## 7. Let's practice!

For now, let's practice these concepts in the exercises.



## Splitting the rental rate

```python
# Get the rental rate column as a string
rental_rate_str = film_df.rental_rate.astype("str")

# Split up and expand the column
rental_rate_expanded = rental_rate_str.str.split(".", expand=True)

# Assign the columns to film_df
film_df = film_df.assign(
    rental_rate_dollar=rental_rate_expanded[0],
    rental_rate_cents=rental_rate_expanded[1],
)
```

## Prepare for transformations

```python
spark.read.jdbc("jdbc:postgresql://localhost:5432/pagila",
                "customer",
                {"user":"repl","password":"password"})
```

## Joining with ratings

```python
# Use groupBy and mean to aggregate the column
ratings_per_film_df = rating_df.groupby('film_id').mean('rating')

# Join the tables using the film_id column
film_df_with_ratings = film_df.join(
    ratings_per_film_df,
    film_df.film_id==ratings_per_film_df.film_id
)

# Show the 5 first results
print(film_df_with_ratings.show(5))
```









# Loading

## 1. Loading

In this video, we'll cover the last step in the ETL process: load. At this point, we've extracted and transformed our data. It now makes sense to load the data ready for analytics.

## 2. Analytics or applications databases

As we mentioned earlier, in databases, there's a clear separation between databases for analytics and databases for applications. For example, complex aggregate queries frequently run on analytical databases, so we should optimize them. On the other hand, application databases have lots of transactions per second so we should optimize them for that. This ties in with something we mentioned before. We often optimize application databases for online transaction processing or OLTP. On the other hand, we optimize analytical databases for online analytical processing or OLAP.

## 3. Column- and row-oriented

At this point, it makes sense to come back to row- vs. column-oriented databases. As mentioned in the first video of this chapter, most **application databases are row-oriented**. That means we store data per record, which makes it easy to add new rows in small transactions. For example, in a **row-oriented database, adding a customer record is easy and fast**. In a column-oriented database, we store data per column. There are multiple reasons why this is optimal for analytics. Without getting too technical, you can think of analytical queries to be mostly about a small subset of columns in a table. By storing data per column, it's faster to loop over these specific columns to resolve a query. In a row-oriented system, we would lose time skipping unused columns for each row. Column-oriented databases also lend themselves better to parallelization.

![image-20210323215541809](https://i.loli.net/2021/03/24/azPs6hC8wHdO2fF.png)

## 4. MPP Databases

That brings us seamlessly to a type of database which is often a target at the end of an ETL process. They're called **massively parallel processing databases**. They're column-oriented databases optimized for analytics, that run in a distributed fashion. Specifically, this means that queries are not executed on a single compute node, but rather split into subtasks and distributed among several nodes. Famous managed examples of these are **Amazon Redshift, Azure SQL Data Warehouse, or Google BigQuery.**

## 5. An example: Redshift

Let's look at an example. **To load data into Amazon Redshift, an excellent way to do this would be to write files to S3, AWS's file storage service, and send a copy query to Redshift.** Typically, MPP databases load data best from files that use a columnar storage format. CSV files would not be a good option, for example. We often use a file format called parquet for this purpose. There are helper functions to write this kind of files in several packages. For example, in pandas, you can use the `.to_parquet()` method on a dataframe. In PySpark, you can use `.write.parquet()`. You can then connect to Redshift using a PostgreSQL connection URI and copy the data from S3 into Redshift, like this.

```python
# Pandas .to_parquet() method
df.to_parquet("./s3://path/to/bucket/customer.parquet")
# PySpark .write.parquet() method
df.write.parquet("./s3://path/to/bucket/customer.parquet")
```

## 6. Load to PostgreSQL

In other cases, you might want to load the result of the transformation phase into a PostgreSQL database. For example, your data pipeline could extract from a rating table, transform it to find recommendations and load them into a PostgreSQL database, ready to be used by a recommendation service. For this, there are also several helper methods in popular data science packages. For example, you could use `.to_sql()` in Pandas. Often, you can also provide a strategy for when the table already exists. Valid strategies for `.to_sql()` in Pandas are: "fail", "replace" and "append".

## 7. Let's practice!

That's all for this video. Let's do some exercises.

```python
# Write the pandas DataFrame to parquet
film_pdf.to_parquet("films_pdf.parquet")

# Write the PySpark DataFrame to parquet
film_sdf.write.parquet("films_sdf.parquet")
```

## Load into Postgres

```python
# Finish the connection URI
connection_uri = "postgresql://repl:password@localhost:5432/dwh"
db_engine_dwh = sqlalchemy.create_engine(connection_uri)

# Transformation step, join with recommendations data
film_pdf_joined = film_pdf.join(recommendations)

# Finish the .to_sql() call to write to store.film
film_pdf_joined.to_sql("film", db_engine_dwh, schema="store", if_exists="replace")

# Run the query to fetch the data
pd.read_sql("SELECT film_id, recommended_film_ids FROM store.film", db_engine_dwh)
```



# Putting it all together



## 1. Putting it all together

So we've now covered the full extent of an ETL pipeline. We've extracted data from databases, transformed the data to fit our needs, and loaded them back into a database, the data warehouse. This kind of batched ETL needs to run at a specific moment, and maybe after we completed some other tasks. It's time to put everything together.

## 2. The ETL function

First of all, it's nice to have your ETL behavior encapsulated into a clean `etl()` function. Let's say we have a `extract_table_to_df()` function, which extracts a PostgreSQL table into a pandas DataFrame. Then we could have one or many transformation functions that takes a pandas DataFrame and transform it by putting the data in a more suitable format for analysis. This function could be called `split_columns_transform()`, for example. Last but not least, a `load_df_into_dwh()` function loads the transformed data into a PostgreSQL database. We can define the resulting `etl()` function as follows. The result of `extract_table_to_df()` is used as an input for the transform function. We then use the output of the transform as input for `load_df_into_dwh`.

```python
def extract_table_to_df(tablename, db_engine):
	return pd.read_sql("SELECT * FROM {}".format(tablename), db_engine)
def split_columns_transform(df, column, pat, suffixes):
	# Converts column into str and splits it on pat...
def load_df_into_dwh(film_df, tablename, schema, db_engine):
	return pd.to_sql(tablename, db_engine, schema=schema, if_exists="replace")

db_engines = { ... } # Needs to be configured
def etl():
    # Extract
    film_df = extract_table_to_df("film", db_engines["store"])
    # Transform
    film_df = split_columns_transform(film_df,"rental_rate",".", ["_dollar","_cents"])
    # Load
    load_df_into_dwh(film_df,"film","store", db_engines["dwh"])
```

## 3. Airflow refresher

Now that we have a python function that describes the full ETL, we need to make sure that this function runs at a specific time. Before we go into the specifics, let's look at a small recap of **Airflow**. **Apache Airflow is a workflow scheduler written in Python.** You can represent directed acyclic graphs in Python objects. **DAGs** lend themselves perfectly to manage workflows, as there can be a dependency relation between tasks in the DAG. An operator represents a unit of work in Airflow, and Airflow has many of them built-in. As we saw earlier, you can use a **BashOperator** to run a bash script, for example. There are plenty of other operators as well. Alternatively, you can write a custom operator.

## 4. Scheduling with DAGs in Airflow

So the first thing we need to do is to create the DAG itself. In this code sample, we keep it simple and create a DAG object with id 'sample'. The second argument is `schedule_interval` and it defines when the DAG needs to run. There are multiple ways of defining the interval, but the most common one is using a cron expression. That is a string which represents a set of times. It's a string containing 5 characters, separated by a space. The leftmost character describes minutes, then hours, day of the month, month, and lastly, day of the week. Going into detail would drive us too far, but there are several great resources to learn cron expressions online, for example, the website: https://crontab.guru. The DAG in the code sample run every 0th minute of the hour.

```python
from airflow.models import DAG
dag = DAG(dag_id="sample",...,schedule_interval="0 0 * * *")
```



## 5. The DAG definition file

Having created the DAG, it's time to set the ETL into motion. The etl() function we defined earlier is a Python function, so it makes sense to use the PythonOperator function from the python_operator submodule of airflow. Looking at the documentation of the PythonOperator function, we can see that it expects a Python callable. In our case, this is the `etl()` function we defined before. It also expects two other parameters we're going to pass: `task_id` and `dag`. These are parameters which are standard for all operators. They define the identifier of this task, and the DAG it belongs to. We fill in the DAG we created earlier as a source. We can now set upstream or downstream dependencies between tasks using the `.set_upstream()` or `.set_downstream()` methods. By using `.set_upstream` in the example, `etl_task` will run after `wait_for_this_task` is completed.

```python
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
dag = DAG(dag_id="etl_pipeline",schedule_interval="0 0 * * *")
etl_task = PythonOperator(task_id="etl_task",python_callable=etl,dag=dag)
etl_task.set_upstream(wait_for_this_task)
```



## 6. The DAG definition file

Once you have this DAG definition and some tasks that relate to it, you can write it into a python file and place it in the DAG folder of Airflow. The service detects DAG and shows it in the interface.

```python
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
...
etl_task.set_upstream(wait_for_this_task)
```



## 7. Airflow UI

The Airflow UI will look something like on the following screenshot. Note the task id and schedule interval in the interface.

## 8. Let's practice!

That's all for now. Let's do some exercises.





```python
# Define the ETL function
def etl():
    film_df = extract_film_to_pandas()
    film_df = transform_rental_rate(film_df)
    load_dataframe_to_film(film_df)

# Define the ETL task using PythonOperator
etl_task = PythonOperator(task_id='etl_film',
                          python_callable=etl,
                          dag=dag)

# Set the upstream to wait_for_table and sample run etl()
etl_task.set_upstream(wait_for_table)
etl()
```











# Case Study: DataCamp

# Course ratings

**Got It!**

## 1. Course ratings

Hello again! And welcome to the final chapter of this course. In this chapter, we're going to use everything you've learned thus far and discover how it fits into real-world case studies, by looking at DataCamp's course ratings!

## 2. Ratings at DataCamp

The case study will be about course ratings at DataCamp. As you might have noticed, as a student on DataCamp, you can rate a chapter after you completed it. We can aggregate these chapter ratings to get an estimate of how people rate specific courses. This kind of rating data lends itself to use in recommendation systems.

## 3. Recommend using ratings

You could imagine that top-rated courses are suitable to recommend, for example. Alternatively, you could recommend top-rated Python courses to people that previously rated Python courses highly. There are several ways to go about it, but in essence, we need to get this rating data, clean it where possible, and calculate the top-recommended courses for each user. We could re-calculate this daily, for example, and show the courses in the user's dashboard.

## 4. As an ETL process

In other words, we need to extract rating data, transform it to get useful recommendations, and load it into an application database, ready to be used by several recommendation products. Sounds like a job for the Data Engineer, right? Well... kind of. Usually, this would be a collaboration between a Data Engineer and a Data Scientist. The Data Scientist is responsible for the way recommendations are made, and the Data Engineer fits everything together to get to a stable system that updates recommendations on a schedule. For this case study, we'll describe the full process as a single ETL job. Before we dive into the specifics of the SQL tables we're going to use, it's always nice to look at what we're trying to achieve through a diagram. The extraction happens from two SQL tables in the `datacamp_application` database. It's a PostgreSQL database. The transformation phase consists partly of cleaning up the data, and an algorithm to calculate recommendations from a rating table. Finally, the data needs to be loaded into the `datawarehouse` database, to be used by data products.

## 5. The database

As we've mentioned before, we'll be using two SQL tables from the `datacamp_application` database. The first table is called `courses`. The records are courses in our database. There are four columns we'll look at: `course_id`, which is the internal id we use for courses; `title`, the course title; `description`, a description of the course in English; and `programming_language`: the programming language used in the course. Second, there is a table called `rating` which contains course ratings for all of the courses in the `courses` table. There are three columns in this table: `course_id`, which is a foreign key to the `courses` table; `user_id`, which is the internal identifier of the user that gave the rating; and `rating` which is the one-to-five star rating the user provided for this course.

## 6. The database relationship

The following diagrams could thus clarify the relationship between these tables.

## 7. Let's practice!

Now before we build the recommendation pipeline, it's time to start experimenting with the data in the exercises.



```python

# Complete the connection URI
connection_uri = "postgresql://repl:password@localhost:5432/datacamp_application"
db_engine = sqlalchemy.create_engine(connection_uri)

# Get user with id 4387
user1 = pd.read_sql("SELECT * FROM rating where user_id = 4387", db_engine)

# Get user with id 18163
user2 = pd.read_sql("SELECT * FROM rating where user_id = 18163", db_engine)

# Get user with id 8770
user3 = pd.read_sql("SELECT * FROM rating where user_id = 8770", db_engine)

# Use the helper function to compare the 3 users
print_user_comparison(user1, user2, user3)

```



```python
# Complete the transformation function
def transform_avg_rating(rating_data):
  # Group by course_id and extract average rating per course
  avg_rating = rating_data.groupby('course_id').rating.mean()
  # Return sorted average ratings per course
  sort_rating = avg_rating.sort_values(ascending=False).reset_index()
  return sort_rating

# Extract the rating data into a DataFrame    
rating_data = extract_rating_data(db_engines)

# Use transform_avg_rating on the extracted data and print results
avg_rating_data = transform_avg_rating(rating_data)
print(avg_rating_data) 
```





# From ratings to recommendations

## 1. From ratings to recommendations

Welcome to the second part of this chapter. In the previous exercises, you've seen how to extract and transform data from the rating table. It's time to make our recommendations.

## 2. The recommendations table

The goal is to end up with triplets of data, where the first column is a user id, the second column is a course id, and the final column is a rating prediction. By rating prediction we mean we'll estimate the rating the user would give the course, before they even took it. The triplets form the top three recommended courses for each unique user id in the rating table. Note that this format is useful, as applications that can access the recommendations table can query it for a specific user, and will instantly get three courses to recommend.

## 3. Recommendation techniques

We can use several techniques to transform the rating table into recommendations. Lots of established methods to do this are related to **matrix factorization**. Going into detail on these is not within the scope of this course, but there's an excellent course on 'Building Recommendation Engines with PySpark' in our course catalog at DataCamp.

## 4. Common sense transformation

For this course, let's try to come up with a transformation derived from common sense. The ultimate goal is to take the rating table and extrapolate three courses that would be nice to recommend.

## 5. Average course ratings

In the previous exercises, you've managed to derive average course ratings for each course id. That aggregate will be useful, as we'll want to recommend highly rated courses.

## 6. Use the right programming language

Second, we'll want to recommend courses in the programming language that interests the user. We have data on programming languages of courses, and we have data on which users rate which courses. It makes sense if a user rates 4 courses, 2 of them being SQL, to recommend them a SQL course next. We have all the data to do this in the two tables you already saw.

## 7. Recommend new courses

Finally, we want only to recommend courses that haven't been rated yet by the user. That means that the `user_id` and `course_id` combination in the recommendation should not be in the rating table.

## 8. Our recommendation transformation

Using these three techniques, we can come up with a common-sense recommendation strategy. The first rule is we'll recommend courses in the technologies for which the user has rated most courses. The second rule is we'll not recommend courses that have already been rated for that user. The final rule is we'll recommend the three courses that remain with the highest rating.

## 9. An example

Let's look at an example. A user has rated three courses: two SQL courses and an R course. We'll recommend only SQL courses. We won't recommend courses with id 12 or 52 since the user already rated them. Finally, we'll recommend the three top-rated SQL courses from the remaining SQL courses.

## 10. Let's practice!

In the exercises, we'll write this strategy into a transformation.







```python

course_data = extract_course_data(db_engines)

# Print out the number of missing values per column
print(course_data.isnull().sum())

# The transformation should fill in the missing values
def transform_fill_programming_language(course_data):
    imputed = course_data.fillna({"programming_language": "r"})
    return imputed

transformed = transform_fill_programming_language(course_data)

# Print out the number of missing values per column of transformed
print(transformed.isnull().sum())

```





```python
# Complete the transformation function
def transform_recommendations(avg_course_ratings, courses_to_recommend):
    # Merge both DataFrames
    merged = courses_to_recommend.merge(avg_course_ratings)
    # Sort values by rating and group by user_id
    grouped = merged.sort_values("rating", ascending = False).groupby('user_id') 
    # Produce the top 3 values and sort by user_id
    recommendations = grouped.head(3).sort_values("user_id").reset_index()
    final_recommendations = recommendations[["user_id", "course_id","rating"]]
    # Return final recommendations
    return final_recommendations

# Use the function with the predefined DataFrame objects
recommendations = transform_recommendations(avg_course_ratings, courses_to_recommend)
```





# Scheduling daily jobs



## 1. Scheduling daily jobs

Hi, and welcome to the last lesson of this chapter. In this lesson, we'll see how to do the third and final step of the ETL process.

## 2. What you've done so far

So far, we've seen how to extract data from the `courses` as well as `rating` table using `extract_course_data` and `extract_rating_data`. We've created a function to clean up the missing values in the courses table. We've also created an aggregation function to get the average course ratings per course. You might remember from the previous exercises that we need a DataFrame with average course ratings and a DataFrame with eligible user and course id pairs to get to the recommendations. To get the eligible user and course id pairs, we need to look at the rating table and for each user, generate pairs with the courses that they haven't rated yet. Additionally, we need to look at the courses table and make sure only to recommend courses in the same technology user already showed interest in. The exact implementation would drive us too far, but let's call the function `transform_courses_to_recommend`, and it takes the two tables we extracted as an input. Finally, we already know how to calculate the recommendations with these tables, using transform_recommendations.

![image-20210324005308106](https://i.loli.net/2021/03/24/nJfrPsVXgUiTepc.png)

## 3. Loading to Postgres

Now, it's time to load the data into a Postgres table. In this case, we could use the table in data products like a recommendation engine. Finally, we'll orchestrate everything in an Airflow job to make sure we keep the table up to date daily. That could be essential if we want to send out daily e-mails to specific customers with recommended courses, for example.

## 4. The loading phase

Ok, let's look at the loading phase first. To get the data into the table, we can take the recommendations DataFrame that we've built in a previous exercise, and use the `pandas` `.to_sql` method to write it to a SQL table. We've seen the method in the third chapter, but here's a little refresher on the syntax. It takes the table name as a first argument, a database engine, and finally, we can define a strategy for when the table exists. We could use `"append"` as a value, for example, if we'd like to add records to the database instead of replacing them. This should be enough to get the data into a target database.

```python
recommendations.to_sql("recommendations",db_engine,if_exists="append",)
```

## 5. The etl() function

We now have all pieces to put together to get to the final ETL function to create the recommendations table. We'll start by extracting both the ratings as the courses table from the application's database. We then clean the courses table by filling the NA's, as we did in previous exercises. Afterward, we'll calculate the average course ratings and eligible user and course id pairs. We need these to get to the recommendations using `transform_recommendations()`. We now have everything we need to load the recommendations into Postgres. Here's the full ETL function.

```python
def etl(db_engines):# Extract the data
    courses = extract_course_data(db_engines)
    rating = extract_rating_data(db_engines)
    # Clean up courses data
    courses = transform_fill_programming_language(courses)
    # Get the average course ratings
    avg_course_rating = transform_avg_rating(rating)
	# Get eligible user and course id pairs
    courses_to_recommend = transform_courses_to_recommend(rating,courses,)
    # Calculate the recommendations
	recommendations = transform_recommendations(avg_course_rating,courses_to_recommend,)
    # Load the recommendations into the databaseload_to_dwh(recommendations, db_engine))
```



## 6. Creating the DAG

It's time to wrap things up by creating the final DAG in this course. This time, we'll have to execute the ETL function we just created daily. As we've seen before, in this case, we can use a simple `PythonOperator`. We start by creating the DAG object itself, give it a `schedule_interval` value using the cron notation. We then create a `PythonOperator` and pass the ETL function as callable. We set the DAG of the operator to be the one we just created.

```python
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
dag = DAG(dag_id="recommendations",scheduled_interval="0 0 * * *")
task_recommendations = PythonOperator(task_id="recommendations_task",python_callable=etl,)
```

## 7. Let's practice!

That wraps up the last lesson of this chapter. Have fun doing the exercises.





```python
connection_uri = "postgresql://repl:password@localhost:5432/dwh"
db_engine = sqlalchemy.create_engine(connection_uri)

def load_to_dwh(recommendations):
    recommendations.to_sql("recommendations", db_engine, if_exists="replace")
```



```python
# Define the DAG so it runs on a daily basis
dag = DAG(dag_id="recommendations",
          schedule_interval="0 0 * * *")

# Make sure `etl()` is called in the operator. Pass the correct kwargs.
task_recommendations = PythonOperator(
    task_id="recommendations_task",
    python_callable=etl,
    op_kwargs={"db_engines":db_engines},
)
```



```python
def recommendations_for_user(user_id, threshold=4.5):
  # Join with the courses table
  query = """
  SELECT title, rating FROM recommendations
    INNER JOIN courses ON courses.course_id = recommendations.course_id
    WHERE user_id=%(user_id)s AND rating>%(threshold)s
    ORDER BY rating DESC
  """
  # Add the threshold parameter
  predictions_df = pd.read_sql(query, db_engine, params = {"user_id": user_id, 
                                                           "threshold": threshold})
  return predictions_df.title.values

# Try the function you created
print(recommendations_for_user(12, 4.65))
```



# Congratulations

**Got It!**

## 1. Congratulations

Awesome work. You've finished the course. You're well on your way of becoming a professional data engineer! Let's look at a short recap of what we've learned.

## 2. Introduction to data engineering

In the first chapter, you built a solid understanding of the tasks of a data engineer. You also got acquainted with the different kinds of tools. We finished with some introduction to cloud service providers.

## 3. Data engineering toolbox

In the second chapter, you took a deep dive into the data engineering toolbox. Starting with a lesson about databases, we talked about parallel computing using frameworks like Spark and finished it off with some workflow scheduling in Airflow.

## 4. Extract, Load and Transform (ETL)

The third chapter introduced the concept of ETL, or Extract, Load, and Transform. This well-established procedure consists of the extract phase, where we extract data from several sources. Afterward, we transform it using parallel computing frameworks. Finally, we load the result into a target database.

## 5. Case study: DataCamp

We ended with a hands-on example on recommendations at DataCamp, represented as an ETL task. We first fetched data from multiple sources, transformed it to form recommendations, and loaded it into a database that is ready to use by data products.

## 6. Good job!

I hope you enjoyed this course as much as I did and hope it motivates you to learn more about data engineering. See you soon!
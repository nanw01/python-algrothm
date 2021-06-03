# Summary

## Data Engineer Introduction

**Cloud**

problems：

1. data processing

   Self-host data-center

   1. cover electrical and maintenance costs
   2. Peak vs. quiet moments: hard to optimize

2. data storage 

   1. diaster will strike
   2. need different geographical locations



Storage : AWS S3, Azure Blob Storage, Google Cloud Storage

Computation : AWS EC2, Azure Virtual Machines, Google Compute Engine

Databases : AWS RDS, Azure SQL Database, Google Cloud SQL、



**Structured and unstructured data**

Structured: database schema, relational database

Semi-structured : Json

Unstructured: schemaless, more like files 

**SQL and NoSQL**

SQL: Tables, Database schema, Relational databases

NoSQL: Non-relational databases, structured or unstructured, key-value stores(caching), Document DB(JSON objects)



**Parallel compution frameworks**

**Apache hadoop**

1. HDFS
2. MapReduce
3. Hive
   1. runs in hadoop
   2. structured query language: Hive SQL
   3. Initially Mapreduce, now other tools



**Spark**

1. Resilient distributed datasets(RDD)
   1. Spark relies on them
   2. similar to list of tuples
   3. transformations : .map() or .filter() 
   4. actions: .count() or .first()
2. PySpark
   1. pyhton interface to spark
   2. dataframe abstraction
   3. looks similar to pandas



**Workflow scheduling frameworks**

DAGs : Directed Acyclic Graph

1. Set of nodes
2. Directed edges
3. No cycle

Tools : 

- Linux's cron
- Spotify's Luigi
- Apache Airflow : an example DAG



**Extract**

- From Unstructureed, Flat files
- From Josn : javascript object notation, semi-structured, atomic:number,string,bopolean, null, composite:array,object
- From Web through APIs
- From Database
  - Applications database
    - transactions
    - Inserts or changes
    - OLTP
    - Row-oriented
  - Analytical database
    - OLAP
    - Column-oriented



**Transform**

Kind of transformations

- Selection of attribute
- Translation of code value
- Data validation
- Splitting columns into multiple columns
- Joining from multiple sources





**Loading**

Analytics database

- Aggregate queries
- Online analytical processing(OLAP)
- Column-orientd
- Queries about subset of cilumns
- Parallelizaiton

Applications database

- Lots of transactions
- Online transaction processing
- Row-oriented
- Stored per recoed
- Added per transaction
- E.G. adding customer is fast



Mpp Databases : Massivelly Parallel Processing Databases

- Amazon Redshfit
- Azure SQL Data Warehouse
- Google BigQuery





Data engineering toolbox

- Databases
- Parallel computing & frameworks (Spark)
- Workflow scheduling with Airflow

Extract, Load and Transform(ETL)

- Extract : get data from several sources
- Transform : perform transformations using parallel computing
- Load: load data into target database









## Shell

Curl

![image-20210522192223193](https://tva1.sinaimg.cn/large/008i3skNgy1gqs05iy88jj30uc0aqabd.jpg)

```shell

# check curl installation
man curl
# Basic curl syntax
curl [option flags] [url]

# -O use original name
curl -O https://****.com/datafilename.txt
# -o + new file name, rename the file
curl -o renameddatafilename.txt https://www.****.com/datafilename.txt

# using wildcards
# download every file hosted on link
curl -O http://www.****.com/datafinename*.txt
# Globbing Parser
curl -O http://www.****.com/datafilename[001-100].txt
curl -O http://www.****.com/datafilename[001-100:10].txt

```



![image-20210522192243043](https://tva1.sinaimg.cn/large/008i3skNgy1gqs05uz1pej30s209ygnc.jpg)

```shell
# check Wget installation
which wget
# basic Wget syntax
wget [option flags] [url]
# option flags
# -b go to background immediately after startup
# -q turn off wget output
# -c resume broken download
wget -bqc https://www.****.com/deatfilename.txt
# chakan
cat url_list.txt
# download from the url locations stored within the file url_list.txt using -i
wget -i url_list.txt
# set download constraints
wget --limit-rate={rate}k {file_location}
wget --limit-rate=200k -i url_list.txt
# set a mandatory pause time between file downloads with --wait
wget --wait={second} {file_location}
wget --wait=2.5 -i url_list.txt
```



![image-20210522194831307](https://tva1.sinaimg.cn/large/008i3skNgy1gqs0wq3cz5j312g0egq63.jpg)



What is csvkit?

is a suite of command-line tools

Is developed in python by wireservice

offer data processing and cleaning capabilities on csv files.

has data capabilities that rival python, r, sql

```shell
in2csv --help
in2csv -h
# converting diles to csv
in2csv SpotifData.xlsx > SpotifyData.csv
# --names or -n option to print all sheet names in SpotifyData.xlsx
in2csv -n SpotifyData.xlsx
# convert sheet
in2csv SpotifyData.xlsx --sheet 'Worksheet1_populaty.xlsx' > SpotifyData.csv

# cvslook - read a csv
cvslooc SpotifyData.csv

# csvstat - print descriptive summary
csvstat Spotify_popularity.csv

# csvcut - filters data using column name or position
# csvgrep - filters data by row calue throught exact match, pattern matching, or even regex
csvcut -n Spotify_MusicAttributes.csv
# return the first colun in the data by position
csvcut -c 1 Spotify_MusicAttributes.csv
csvcut -c 'track_id' Spotify_MusicAttributes.csv
# 2,3
csvcut -c 2,3 Spotify_MusicAttributes.csv
csvcut -c 'danceability','duration_ms' Spotify_MusicAttributes.csv

# csvgrep






```









```shell
# sql2csv query against the database
sql2csv --db 'sqlite:///SpotifyDatabase.db' -- query 'select * from Spotify_popularity' > Spotify_Popularity.csv

# apply sql to a local csv file
csvsql --query 'select * from spotify_musicattributes limit 1' Spotify_MusicAttributes.csv
# apply sql to a local csv file and print
csvsql --query 'select * from spotify_musicattributes limit 1' data/Spotify_MusicAttributes.csv | csvlook
# save local
csvsql --query 'select * from spotify_musicattributes limit 1' data/Spotify_MusicAttributes.csv > OneSongFile.csv

# join csvs using sql syntax
csvsql --query 'select *  from dile_a inner join file_b' file_a.csv file_b.csv


# csv --insert --db --no-inference & --no_constraints
csvsql --db "sqlite:///SpotifyDatabase.db" --insert Spotify_MusicAttributes.csv
# --no-inference disable type inference when parsing the input
# --no-constraints generate a schema without length limits or null check
csvsql --no-inference --no-constraints  --db "sqlite:///SpotifyDatabase.db" --insert Spotify_MusicAttributes.csv


```



Python on the command line

```python

#
echo "print('Hello World!')" > hello_world.py

# pip upgrade
pip install --upgrade pip

```



Scheduler 

Scheduler runs jobs ona pre-detemined schedule.,  Airflow, Luigi, Rundeck, etc.

Corn

1. a time-based job-schduler
2. comes pre-installed in MacOS,Unix
3. can be installed in windows via cygwin or replaced with windows task scheduler
4. is used to automate jobs like system maintenace, bash scripts, python jobs.





```shell
# crontab : is a central file to keep track of cron jobs.
Crontab -l

#
 echo "* * * * * python create_model.py" | crontab
 
.---------------- minute (0 - 59)
| .------------- hour (0 - 23)
| | .---------- day of month (1 - 31)
| | | .------- month (1 - 12) OR jan,feb,mar,apr ...
| | | | .---- day of week (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed ... 
| | | | |
* * * * * command-to-be-executed
 
```







## Airflow

what is workflow?

A workflow is a set of steps to accomplish a given data engineering task.

What is airflow?

Airflow is a platform to program workflows (general), including the **creation**, **scheduling**, and **monitoring** of said workflows.



Airflow implements workflows as DAGs, or Directed Acyclic Graphs.

Airflow can be accessed and controlled via code, via the command-line, or via a built-in web interface. 

what is DAGs?

A DAG stands for a Directed Acyclic Graph. this represents the set of tasks that make up your workflow. It consists of the tasks and the dependencies between tasks.

Using thee Airflow run shell command.  Airflow run takes three arguments, a dag_id, a task_id, and a start_date.

```shell
airflow run <dag_id> <task_id> <start_date>
```



Directed : inhere flow representing the dependencies or order between execution of components.

Acyclic: not loop or repeat

Graph: represents the components and the relationshops or dependeenciees between them



Airflow DAGs are made up of components to be executed, such as operators, sensors, etc. Airflow typically refers to these as tasks. 



Define a DAG,

1.first import the DAG object from airflow dot models

2.create a default arguments dictionary consisting of attributes that will be applied to the components of our DAG. such as , owner, emial, start_date.

3.define our DAG object with the first argument using a name for the DAG, etl underscore workflow, and assign the default arguments dictionary to the default underscore args argument



![image-20210324202525022](https://tva1.sinaimg.cn/large/008i3skNgy1gqru9oq7woj31eo0aqac5.jpg)





**Airflow operators**

AirFlow operators : The most common task in Airflow is the Operator. Airflow operators represent a single task in a workflow.

running a command, sending an email, running a Python script

run independently , Generally, do not share information between each other but it is possible



Airflow contains many various operators to perform different tasks.

**DummyOperator** can be used to represent a task for troubleshooting or a task that has not yet been implemented

**BashOperator** executes a given Bash command or script. requires three arguments: the task id which is the name that shows up in the UI, the bash command (the raw command or script), and the dag it belongs to.

The **PythonOperator** is similar to the BashOperator, except that it runs a Python function or callable method. 

**EmailOperator**, which as expected sends an email from within an Airflow task. It can contain the typical components of an email, including HTML content and attachments. Note that the Airflow system must be configured with the email server details to successfully send a message.

**sensor operator**, A sensor is a special kind of operator that waits for a certain condition to be true. Some examples of conditions include waiting for the creation of a file, uploading a database record, or a specific response from a web request. With sensors, you can define how often to check for the condition(s) to be true.

**BranchPythonOperator**, Branching provides the ability for conditional logic within Airflow. Basically, this means that tasks can be selectively executed or skipped depending on the result of an Operator. 





**Task**

Within Airflow, tasks are **instantiated** operators. 

Task dependencies in Airflow define an order of task completion. If task dependencies are not defined, task execution is handled by Airflow itself with no guarantees of order.

Task dependencies are referred to as upstream or downstream tasks.

Upstream or Downstream operator. upstream means before and downstream means after.  **bitshift operators**

```python
# Define the tasks
task1 = BashOperator(task_id='first_task',bash_command='echo 1',dag=example_dag)
task2 = BashOperator(task_id='second_task',bash_command='echo 2',dag=example_dag)
# Set first_task to run before second_task
task1 >> task2 # or task2 << task1
```



The PythonOperator is similar to the BashOperator, except that it runs a Python function or callable method.    op_kwargs : op underscore kwargs

```python
def sleep(length_of_time):
    time.sleep(length_of_time)
    
sleep_task = PythonOperator(
    task_id='sleep',
    python_callable=sleep,
    op_kwargs={'length_of_time': 5}
	dag=example_dag
)
```



**airflow.operators.email_operator**



EmailOperator, which as expected sends an email from within an Airflow task. It can contain the typical components of an email, including HTML content and attachments. Note that the Airflow system must be configured with the email server details to successfully send a message.





**Airflow scheduling**

A DAG can be run manually, or via the schedule interval parameter passed when the DAG is defined. Each DAG run maintains a state for itself and the tasks within. The DAGs can have a runing, failed, or success state.

![image-20210324212628714](https://tva1.sinaimg.cn/large/008i3skNgy1gqrvegoy3nj31gn0exq6p.jpg)

![image-20210324212655772](https://tva1.sinaimg.cn/large/008i3skNgy1gqrvezibv9j316b0brgno.jpg)

The cron syntax is the same as the format for scheduling jobs using the Unix cron tool. It consists of five fields separated by a space, starting with the minute value (0 through 59), the hour (0 through 23), the day of the month (1 through 31), the month (1 through 12), and the day of week (0 through 6). 

![image-20210324212646513](https://tva1.sinaimg.cn/large/008i3skNgy1gqrvfxmfnhj31ci0cs0w9.jpg)

```python
# Update the scheduling arguments as defined
default_args = {
  'owner': 'Engineering',
  'start_date': datetime(2019, 11, 1),
  'email': ['airflowresults@datacamp.com'],
  'email_on_failure': False,
  'email_on_retry': False,
  'retries': 3,
  'retry_delay': timedelta(minutes=20)
}

dag = DAG('update_dataflows', default_args=default_args, schedule_interval='30 12 * * 3')


```





**Airflow sensors**

A sensor is a special kind of operator that waits for a certain condition to be true. Some examples of conditions include waiting for the creation of a file, uploading a database record, or a specific response from a web request. With sensors, you can define how often to check for the condition(s) to be true.

default arguments available to all sensors,

**mode**: tells the seensor how to check for the conditon. and has two options, poke or reschedule. The default is **poke**, and means to continue checking until complete without giving up a worker slot. **Reschedule** means to give up the worker slot and wait for another slot to become available.

**poke_interval**: is used in the poke mode, and tells Airflow how often to check for the condition.

timeout:  is how long to wait (in seconds) before marking the sensor task as failed.



**FileSensor** : The FileSensor checks for the existence of a file at a certain location in the file system.

```python
from airflow.contrib.sensors.file_sensor import FileSensor
file_sensor_task = FileSensor(
    task_id='file_sense',
    filepath='salesdata.csv',
    poke_interval=300,
    dag=sales_report_dag
)
init_sales_cleanup >> file_sensor_task >> generate_report
```



**ExternalTaskSensor** waits for a task in a separate DAG to complete.

**HttpSensor** will request a web URL and allow you define the content to check for. 

**SqlSensor** runs a SQL query to check for content.



Why sensors?

1. You're uncertain when a condition will be true.
2. If you want to continue to check for a condition but not necessarily fail the entire DAG immediately.
3. if you want to repeatedly run a check without adding cycles to your DAG, sensors are a good choice.



**executor** 

In Airflow, an executor is the component that actually runs the tasks defined within your workflows. Each executor has different capabilities and behaviors for running the set of tasks.

**SequentialExecutor**: default execution engine, it runs only a single task at a time. Very functionsl fot learning and testing, not recommended for production.

**LocalExecutor** : runs entirely on a single system. It basically treats each task as a process on the local system, and is able to start as many concurrent tasks as desired / requested / and permitted by the system resources. it is defined by the user in one of two ways - either unlimited, or limited to a certain number of simultaneous tasks.

**CeleryExecutor** ：Using a CeleryExecutor, multiple Airflow systems can be configured as workers for a given set of workflows / tasks. 

Celery is a general queuing system written in Python that allows multiple systems to communicate as a basic cluster.

![image-20210324215417463](https://tva1.sinaimg.cn/large/008i3skNgy1gqryb9r8uyj31r20do423.jpg)



**SLAs**

SLA stands for Service Level Agreement.



An SLA miss is any situation where a task or DAG does not meet the expected timing for the SLA. If an SLA is missed, an email alert is sent out per the system configuration and a note is made in the log. 

```python
# 'sla'
task1 = BashOperator(
  task_id='sla_task',
  bash_command='runcode.sh',
  sla=timedelta(seconds=30),
  dag=dag
)

default_args={
 'sla': timedelta(minutes=20)
 'start_date': datetime(2020,2,20)
}
dag = DAG('sla_dag', default_args=default_args)
```



**Reporting**

For reporting purposes you can use email alerting built into Airflow. 

1. Airflow has built-in options for sending messages on success, failure, or error / retry.
2. EmailOperator 



**Building production piplines**



Templates : allow substitution of information during a DAG run. Templates provide added flexibility when defining tasks.

```python
templated_command="""
{% for filename in params.filenames %}
  echo "Reading {{ filename }}"
{% endfor %}
"""
t1 = BashOperator(task_id='template_task',
       bash_command=templated_command,
       params={'filenames': ['file1.txt', 'file2.txt']}
       dag=example_dag)
```

```python
# Macros
from airflow.models import DAG
from airflow.operators.email_operator import EmailOperator
from datetime import datetime

# Create the string representing the html email content
html_email_str = """
Date: {{ ds }}
Username: {{ params.username }}
"""

email_dag = DAG('template_email_test',
                default_args={'start_date': datetime(2020, 4, 15)},
                schedule_interval='@weekly')
                
email_task = EmailOperator(task_id='email_task',
                           to='testuser@datacamp.com',
                           subject="{{ macros.uuid.uuid4() }}",
                           html_content=html_email_str,
                           params={'username': 'testemailuser'},
                           dag=email_dag)


```





**Branching** : provides the ability for conditional logic within Airflow. Basically, this means that tasks can be selectively executed or skipped depending on the result of an Operator.



```python
 def branch_test(**kwargs):
  if int(kwargs['ds_nodash']) % 2 == 0:
    return 'even_day_task'
  else:
    return 'odd_day_task'
branch_task = BranchPythonOperator(task_id='branch_task',dag=dag,
       provide_context=True,
       python_callable=branch_test)
start_task >> branch_task >> even_day_task >> even_day_task2
branch_task >> odd_day_task >> odd_day_task2
```









Creating a production pipeline







```bash
airflow test etl_update sense_file -1
```



```python
from airflow.models import DAG
from airflow.contrib.sensors.file_sensor import FileSensor

# Import the needed operators
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import date, datetime

def process_data(**context):
  file = open('/home/repl/workspace/processed_data.tmp', 'w')
  file.write(f'Data processed on {date.today()}')
  file.close()

    
dag = DAG(dag_id='etl_update', default_args={'start_date': datetime(2020,4,1)})

sensor = FileSensor(task_id='sense_file', 
                    filepath='/home/repl/workspace/startprocess.txt',
                    poke_interval=5,
                    timeout=15,
                    dag=dag)

bash_task = BashOperator(task_id='cleanup_tempfiles', 
                         bash_command='rm -f /home/repl/*.tmp',
                         dag=dag)

python_task = PythonOperator(task_id='run_processing', 
                             python_callable=process_data,
                             dag=dag)

sensor >> bash_task >> python_task

```

```python
from airflow.models import DAG
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from dags.process import process_data
from datetime import timedelta, datetime

# Update the default arguments and apply them to the DAG
default_args = {
  'start_date': datetime(2019,1,1),
  'sla' : timedelta(minutes=90)
}

dag = DAG(dag_id='etl_update', default_args=default_args)

sensor = FileSensor(task_id='sense_file', 
                    filepath='/home/repl/workspace/startprocess.txt',
                    poke_interval=45,
                    dag=dag)

bash_task = BashOperator(task_id='cleanup_tempfiles', 
                         bash_command='rm -f /home/repl/*.tmp',
                         dag=dag)

python_task = PythonOperator(task_id='run_processing', 
                             python_callable=process_data,
                             provide_context=True,
                             dag=dag)

sensor >> bash_task >> python_task

```



```python
from airflow.models import DAG
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from dags.process import process_data
from datetime import datetime, timedelta

# Update the default arguments and apply them to the DAG.

default_args = {
  'start_date': datetime(2019,1,1),
  'sla': timedelta(minutes=90)
}
    
dag = DAG(dag_id='etl_update', default_args=default_args)

sensor = FileSensor(task_id='sense_file', 
                    filepath='/home/repl/workspace/startprocess.txt',
                    poke_interval=45,
                    dag=dag)

bash_task = BashOperator(task_id='cleanup_tempfiles', 
                         bash_command='rm -f /home/repl/*.tmp',
                         dag=dag)

python_task = PythonOperator(task_id='run_processing', 
                             python_callable=process_data,
                             provide_context=True,
                             dag=dag)


email_subject="""
  Email report for {{ params.department }} on {{ ds_nodash }}
"""


email_report_task = EmailOperator(task_id='email_report_task',
                                  to='sales@mycompany.com',
                                  subject=email_subject,
                                  html_content='',
                                  params={'department': 'Data subscription services'},
                                  dag=dag)


no_email_task = DummyOperator(task_id='no_email_task', dag=dag)


def check_weekend(**kwargs):
    dt = datetime.strptime(kwargs['execution_date'],"%Y-%m-%d")
    # If dt.weekday() is 0-4, it's Monday - Friday. If 5 or 6, it's Sat / Sun.
    if (dt.weekday() < 5):
        return 'email_report_task'
    else:
        return 'no_email_task'
    
    
branch_task = BranchPythonOperator(task_id='check_if_weekend',
                                   python_callable=check_weekend,
                                   provide_context=True,
                                   dag=dag)

    
sensor >> bash_task >> python_task

python_task >> branch_task >> [email_report_task, no_email_task]

```



**What we've learned**

Let's review everything we've worked with during this course. We started with learning about workflows and DAGs in Airflow. We've learned what an operator is and how to use several of the available ones. We learned about tasks and how they are defined by various types of operators. In addition, we learned about dependencies between tasks and how to set them with bitshift operators. We've used sensors to react to workflow conditions and state. We've scheduled DAGs in various ways. We used SLAs and alerting to maintain visibility on our workflows. We learned about the power of templating in our workflows for maximum flexibility when defining tasks. We've learned how to use branching to add conditional logic to our DAGs. Finally, we've learned about the Airflow interfaces (command line and UI), about Airflow executors, and a bit about how to debug and troubleshoot various issues with Airflow and our own workflows.







## DataBase

### database design

- shcemas : logically organized?
- Normalization : minimal dependency and redundancy?
- Views : Joins?
- Access control 
- DBMS : pick between all the SQL and noSQL options?





|         | OLTP                                   | OLAP                                          |
| ------- | -------------------------------------- | --------------------------------------------- |
| Purpose | support daily transaction              | Report and analyze data                       |
| Design  | Application-oriented                   | Subject-oriented                              |
| Data    | Up-to-date, operational                | Consolidated, historical                      |
| Size    | Snapshot, gigabytes                    | Archive, terabytes                            |
| Queries | Simple transactions & frequent updates | Complex, aggregate queriees & limited updates |
| Users   | Thousands                              | hundres                                       |



| Structured data                     | Unstructured data                  | Semi-structured data          |
| ----------------------------------- | ---------------------------------- | ----------------------------- |
| Follows a schema                    | Schemaless                         | does not follow larger schema |
| Define data type & relationshops    | Makes up most of data in the world | Self-describing structure     |
| SQL,tables in a realtional database | Photos, chat logs, MP3             | NoSQL,XML,JSON                |

- 1) Defined vs Undefined Data 
- 2) Qualitative vs Quantitative Data
- 3) Storage in Data Houses vs Data Lakes
- 4) Ease of Analysis
- 5) Predefined Format vs Variety of Formats



A **relational database** is a type of **database** that stores and provides access to data points that are related to one another.







**Data warehouses**

- Optimized for analytics - OLAP
  - Organized for reading/aggregating data
  - Usually read-only
- Contains data from multiple sources
- **Massively Parallel Processing** (MPP)
- Typically uses a denormalized schema and dimensional modeling

**Data marts**

- Subset of data warehouses
- Dedicated to a specific topic

**Data lakes**

- Store all types of data at a lower cost
  - e.g., raw, operational databases, loT device logs, real-time, relational and non-relational
- Retains all data and can take up petabytes
- Schema-on-read as pposed to schema-on-write
- need to catalog data otherwise becomes a data swamp
- run big data analytics using services such as Apache Spark and Hadoop
  - Useful for deep learning and ata discovery because activities require so much data



![image-20210523145033875](https://tva1.sinaimg.cn/large/008i3skNgy1gqsxx1l473j314s0la0xk.jpg)



Data modeling : process of creating a data model for the data to be stored



A data lake is a vast pool of raw data, the purpose for which is not yet defined. 

A data warehouse is a repository for structured, filtered data that has already been processed for a specific purpose.



|                 | **Data Lake**                         | **Data Warehouse**                          |
| --------------- | ------------------------------------- | ------------------------------------------- |
| Data Structure  | Raw                                   | Processed                                   |
| Purpose of Data | Not Yet Determined                    | Currently In Use                            |
| Users           | Data Scientists                       | Business Professionals                      |
| Accessibility   | Highly accessible and quick to update | More complicated and costly to make changes |







**Data Models**

[Introduction to Data Modelling. What is Data Modelling? | by Sagar Lad | Sagar Explains Azure and Analytics : Data Engineering Series | Medium](https://medium.com/sagar-explains-azure-and-analytics-data-engineerin/introduction-to-data-modelling-c0c44432ec0b#:~:text=A data model helps design,to create a physical database.)

1. Conceptual data model

   describes entities, relationship, and attributes

2. Logical data model

   defines tables, cilumns, relationships

3. Physical data model

   Describe physical storage



**Dimensional modeling**

*Dimensional Modeling* (DM) is a data structure technique optimized for data storage in a Data warehouse.



There are two types of design schemas in data modeling, which are: Snowflake schema and Star schema.



**Star schema**

Dimensional  modeling: star schema

Fact tables

- Holds records of a metric
- Changes regularly
- Connects to dimensions via foreign keys

Demension tables

- Holds descriptioons of attributes
- Does not change as often

Fact Table:

- Fact is an atomic presentation of certain business measurements (dollars, counts etc.)

- Fact table’s atomic level (most granular level) is defined by the combination of dimension keys.

Dimension Table:

- Dimensions are “things” that are used to slice the data.

- Dimensions are usually organized in hierarchies of categories, levels and members.

- Dimensions can be shared with multiple facts to provide correlation.

- Dimension tables are usually carry a “surrogate key” as the primary key.

- Dimension tables usually has a multi-part business key.

- Dimension table itself can be further normalized to provide higher level of granularity and flexibility.

- Changing dimensions:
  - 0 – no change
  - 1 – overwrite old value
  - 2 – add new value while keep history
  - 3 – add new attributes to the dimension





![image-20210523153438798](https://tva1.sinaimg.cn/large/008i3skNgy1gqsz6vs4ndj31820j4ago.jpg)

![image-20210523153506949](https://tva1.sinaimg.cn/large/008i3skNgy1gqsz7cyscaj311a0j4n1f.jpg)



Same fact table, different dimensions

Star schemas : one dimension

Snowflake schemas : more than one dimension. Because dimension tables are normalized.



Normalization

Normal forms(NF)

Data anomalies : if **not** normalize enough

1. Updata anomaly
2. Insertion anomaly
3. Deletion anomaly



- 3NF is designed to provide consistent write performances to support OLTP application.

- 3NF is designed to a single application. The access path of data is usually not optimized for range scans and aggregations

- 3NF is designed in such way that business analytics need to perform multiple joins to represent a single business concept.

- 3NF is designed to present a snapshot of business state or process while no history is provided nor does business process itself can be clearly presented.



**Dimensional Modeling (DM)** is a data structure technique optimized for data storage in a Data warehouse. 











**Database Views**

Views are **virtual tables** that can be a great way to optimize your database experience. Virtual table that is not part od the physical schema

Benefits of view

- does't take up storage

- A form of access control
  - Hide sensitive columns and restrict what user can see
- Masks complexity of queried
  - Userful for highly normalized schemas

Managing views

- Aggregation: SUM(), AVG(), COUNT(), MIN(), MAX(), GROUP BY, etc
- Joins: INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN
- Conditinals: WHERE, HAVING, UNIQUE, NOT NULL, AND, OR,  >, < , etc



```sql
-- Granting and revokeing access to a view

GRANT privilege(s) or REVOKE privilege(s)
ON object
TO role or FROM role
```

- Privileges: SELECT, INSERT, UPDATE, DELETE, etc

- Objects: table, view, schema
- Roles: a database user or a group of database users





Updating a view

Inserting into a view

- Not all views are insertable

- Avoid  modifying data through view

Dropping a view

Redefining a view

Altering a view





Materialized views



Views

- known as non-materialized view

Materialized views

- Stores the  query results, not the query
- querying a materialized view means accessing the stored query results
  - not running the query like a non-materialized view
- Refreshed or rematerilized when prompted or scheduled



When to use materialized view

- Long running queries
- Underlying query results don't change often
- Data warehouses  because OLAP is not writ-intensive
  - Save on computational cost of frequent queries



Managing dependencies

- Materialized views often depend onother  materialized views

- Creates a dependency chain when refreshing views
- Not the most efficient to refresh all view at the same time



Tools for managing dependencies

- Use Directed Acyclic Graphs (DAGs) to keep track of view
- Pipline scheduler tools



**Database roles and access control**

```sql
-- Granting and revokeing access to a view

GRANT privilege(s) or REVOKE privilege(s)
ON object
TO role or FROM role
```

- Privileges: SELECT, INSERT, UPDATE, DELETE, etc

- Objects: table, view, schema
- Roles: a database user or a group of database users



Database roles

- Manage role is an entity access permission
- A database role is an entity thay contains information that
  - Define the role's privileges
  - Interact with the client authentication system
- Roles can be assigned to one or more users
- Roles are global access a database cluster installation





Create a role

```sql
-- Empty role
CREATE ROLE data_analyst;
-- Roles with some a ributes set
CREATE ROLE intern WITH PASSWORD 'PasswordForIntern' VALID UNTIL '2020-01-01';
CREATE ROLE admin CREATEDB;
ALTER ROLE admin CREATEROLE;
```



GRANT and REVOKE privileges from roles

```sql
GRANT UPDATE ON ratings TO data_analyst;
REVOKE UPDATE ON ratings FROM data_analyst;
```

The available privileges in PostgreSQL are:

SELECT , INSERT , UPDATE , DELETE , TRUNCATE , REFERENCES , TRIGGER , CREATE , CONNECT , TEMPORARY , EXECUTE , and USAGE



Role :

- User roles
- Group roles

```sql
-- Group role
CREATE ROLE data_analyst;
-- User role
CREATE ROLE intern WITH PASSWORD 'PasswordForIntern' VALID UNTIL '2020-01-01';
```



```sql
-- Group role
CREATE ROLE data_analyst;
-- User role
CREATE ROLE alex WITH PASSWORD 'PasswordForIntern' VALID UNTIL
'2020-01-01' ;
GRANT data_analyst TO alex;
REVOKE data_analyst  FROM alex;
```



Benefits

- Roles live on a/er users are deleted

- Roles can be created before user accounts

- Save DBAs time

Pitfalls

- Sometimes a role gives a specific user too much access

- You need to pay a ention







**Table Partitioning**

Why

Problem: queries/updates become slower

Because: indices don't fit memory

Solution: split table into smaller parts (=partitioning)



Data modeling

1. Conceptual data model

2. Logical data model

   For partitioning, logical data model is the same

3. Physical data model

   Partitioning is part of physical data model



Vertical partitioning

Horizontal partitioning





Pros/cons of horizontal partitioning

Pros

- Indices of heavily-userd partitions fit in memory
- Move to specific medium: slower vs. faster
- Used for both OLAP as OLTP

Cons

- Patitioning existing table can be a hassle
- Some constraints can not be set





**Data Integration** combines data from diferent sources, formats, technologies to provide users with a translated and unified view of that data.

Transformations



Choosing a data integration tool

- Flexible
- Reliable
- Scalable





**Database Management System (DBMS)** is a software that is used to define, create and maintain a database and provides controlled access to the data.

**Relational Database Management System (RDBMS)** is an advanced version of a DBMS.



**Database  Management System (DBMS)**

- DBMS: Database Management System

- Create and maintain databases
  - Data
  - Database schema
  - Database engine
- Interface between database and end users



DBMS types

- Choice of DBMS depends on database type
- Two types:
  - SQL DBMS
  - NoSQL DBMS







![image-20210523200040641](https://tva1.sinaimg.cn/large/008i3skNgy1gqt6votfjdj30yq0jymz7.jpg)

**SQL DBMS**

- Relational DataBase Management System(RDBMS)

- Based on the relational model of data

- Query language: SQL

- Best option when:
  - Data is structured and unchanging
  - Data must be consistent



**NoSQL DBMS**

- Less structured

- Document-centered rather than table-centered

- Data doesn’t have to fit into well-defined rows and columns

- Best option when:
  - Rapid growth
  - No clear schema de,nitions
  - Large quantities of data

- Types: key-value store, document store, columnar database, graph database



NoSQL DBMS - **key-value store**

- Combinations of keys and values
  - Key: unique identifier
  - Value: anything

- Use case: managing the shopping cart for
- an on-line buyer

- Example: **redis**

NoSQL DBMS - **document store**

- Similar to key-value
- Values (= documents) are structured
- Use case: content management
- Example: **mongoDB**

NoSQL DBMS - **Columnar database**

- Store data in columns
- Scalable
- Use case: big data analytics where speed is important
- Example: **cassandra**



NoSQL DBMS - **graph databae**

Data is interconnected and best

represented as a graph

Use case: social media data, recommendations

Example: **neo4j**









**Heap Table** - Traditional Design Approach of **OLTP** Database

- Insert efficiently
- not efficient in term of searching for a particular record or a group of records



Optimize read on Heap Table -

1. **Index**

   An ***index*** is a copy of selected columns of data, from a table, that is designed to enable very efficient search. 

   - To minimize the table scan to search.

   - Rowid is a relative position in the table file. Use rowid software can directly access the record without scanning the file

2. **Parallel**

   Read table by multiple thread/process can improve efficiency

   Bottleneck in this approach is slowest linke-from disk to cpu.



**Massively Parallel Processing**

[What is an MPP Database? Intro to Massively Parallel Processing | FlyData | Real Time MySQL Replication to Amazon Redshift](https://www.flydata.com/blog/introduction-to-massively-parallel-processing/)

**Benefits** of MPP systems

- Do not reply on single disk IO system to provide parallel throughput. *disk io sub system are usually expensive
- Provide parallel computing in a easy model so that all CPUs working independently and provide very high aggregated throughput.
- Relatively easy to provide N factor throughput by adding nodes.

**Disadvantages** of MPP system

- High latency on network link
- Susceptible to component failure.
- Transaction control and locks need coordinates among the nodes which relies on high latency network interconnect. Hence not suited for high volume OLTP application.





Teradata

- A Relational Database Management Syatem



CREATE TABLE

DROP TABLE

CREATE VIEW

DROP VIEW



A **database** is an organized collection of data, generally stored and accessed electronically from a computer system. 

A **table** is a collection of related data held in a table format within a database.

A **volatile table** is a temporary **table** that is only held until the end of session.

**Global temporary tables** have a persistent definition but do not have persistent contents across sessions. 

A **primary key** is a field in a table which uniquely identifies each row/record in a *database* table.

A **foreign key** is a set of attributes in a table that refers to the primary key of another table. 





**Teradata**

- Unlike traditional “single computer”databases, Teradata is build on multiple computers connected by network. We call this MPP system.

- Slowest operation in MPP system is move data across network.

- Data movement in Teradata is called“data redistribution”.



**Teradata BTEQ** stands for Basic **Teradata** Query. I











## Spark - Hadoop

Big data: Volume, Variety and Velocity

Big Data concepts and Terminology

- Clustered computing: Collection of resources of multiple machines
- Parallel computing: Simultaneous computation
- Distributed computing: Collection of nodes (networked computers) that run in parallel
- Batch processing: Breaking the job into small pieces and running them on individual machines
- Real-time processing: Immediate processing of data



Processing Systems

**Hadoop/MapReduce**: Scalable and fault tolerant framework wri en in Java

- Open source
- Batch processing

**Apache Spark:** General purpose and lightning fast cluster computing system

- Open source
- Both batch and real-time data processing



**Features**

- Cluster computing
- Efficent in-memory computation for large data sets
- Lighting fast data processing framework
- Provides  support for java, scala, python,R, and SQL



**Spark Components:**

- Spark SQL
- MLlib
- GraphX
- Spark Streaming



Spark modes of deployment

- Local model: single machine such as your laptop
  - convennient for  testing, debugging and demonstration
- Cluster model: Set of pre defined machines
  - Good for production
- Workflow: local -> clusters
- No code change necessary



**Spark shell**

- Interactive environment for running Spark jobs
- Helpful for fast interactive prototyping
- Spark’s shells allow interacting with data on disk or in memory



**SparkContext**

- An entry point into the world of Spark
- An entry point is a way of connecting to Spark cluster
- An entry point is like a key to the house



**RDD**

Resilient Distributed Datasets

- Resilient: Ability to withstand failures
- Distributed: Spanning across multiple machines
- Datasets: Collection of partitioned data e.g, Arrays, Tables, Tuples etc.,



**PySpark operations**

- **Transformations** : **create** new RDDs
  - map()
  - filter()
  - flatMap()
  - union()

- **Actions** : perform **conputation** on the RDDs
  - collect()
  - take(N)
  - first()
  - count()



**Pair RDDs**

- Two common ways to create pair RDDs
  - From a list of key-value tuple
  - From a regular RDD

Pair RDDs Transformations

- all regular transformations
  - map()
  - filter()
  - flatMap()
  - union()
- paried RDD Transformations (poperate onkey value pairs)
  - reduceByKey(func): Combine values with the same key
  - groupByKey(): Group values with the same key
  - sortByKey(): Return an RDD sorted by the key
  - join(): Join two pair RDDs based on their key







MapReduce流程：input->Splitting->Mapping->Shuffling->Reducing-> result

![image-20210524173153342](https://tva1.sinaimg.cn/large/008i3skNgy1gqu87a4mqkj315o0ju4fc.jpg)





**Actions**

- reduce()
- saveAsTextFile()
- countByKey()
- collectAsMap()





**PySpark DataFrames**

- PySpark SQL is a Spark library for structured data. It provides more information about the structure of data and computation
- PySpark DataFrame is an immutable distributed collection of data with named columns
- Designed for processing both structured (e.g relational database) and semi-structured data (e.g JSON)
- Dataframe API is available in Python, R, Scala, and Java
- DataFrames in PySpark support both SQL queries ( SELECT * from table ) or expression methods ( df.select() )





SparkContext is the main entry point for creating RDDs



SparkSession provides a single point of entry to interact with Spark DataFrames

SparkSession is used to create DataFrame, register DataFrames, execute SQL queries

SparkSession is available in PySpark shell as spark





DataFrame operator

- Transformations

  - select(), ,lter(), groupby(), orderby(), dropDuplicates() and withColumnRenamed()

- Actions

  - printSchema(), head(), show(), count(), columns and describe()

  





DataFrame API vs SQL queries

- In PySpark You can interact with SparkSQL through DataFrame API and SQL queries
- The DataFrame API provides a programmatic domain-speci,c language (**DSL**) for data
- DataFrame transformations and actions are **easier to construct programmatically**
- SQL queries can be concise and **easier to understand** and **portable**
- The operations on DataFrames can also be done using SQL queries



**Data Visualization**

Plotting graphs

- pyspark_dist_explore library
- toPandas()
- HandySpark library





Pandas DataFrame vs PySpark DataFrame

- Pandas DataFrames are in-memory, single-server based structures and operations on PySpark run in parallel
- The result is generated as we apply any operation in Pandas whereas operations in PySpark
- DataFrame are lazy evaluation
- Pandas DataFrame as mutable and PySpark DataFrames are immutable
- Pandas API support more operations than PySpark Dataframe API



HandySpark method of visualization

**HandySpark** is a package designed to improve PySpark user experience







**PySpark MLlib** is a component of Apache Spark for machine learning

ML Algorithms

- Collaborative filtering
- Classification
- Clustering 

Featurization

- Feature extraction
- transformation
- Dimensionality reduction
- Selection 

Piplines:

- tools fpr constructing
- Evaluating 
- Tuning ML Pipeline





Why PySpark MLlib?

- Scikit-learn is a popular Python library for data mining and machine learning
- Scikit-learn algorithms **only work for small datasets on a single machine**
- Spark's MLlib algorithms are **designed for parallel processing on a cluster**
- Supports languages such as Scala, Java, and R
- Provides a **high-level API** to build machine learning pipelines





PySpark MLlib Algorithms

- Classification (Binary and Multiclass) and Regression: 
  - Linear SVMs, 
  - logistic regression,
  - decision trees, 
  - random forests, 
  - gradient-boosted trees, 
  - naive Bayes, 
  - linear least squares,
  - Lasso, 
  - ridge regression, 
  - isotonic regression
- Collaborative filtering: 
  - Alternating least squares (ALS)
- Clustering: 
  - K-means, 
  - Gaussian mixture, 
  - Bisecting K-means and 
  - Streaming K-Means

```python
# pyspark.mllib.recommendation
from pyspark.mllib.recommendation import ALS
# pyspark.mllib.classification
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# pyspark.mllib.clustering
from pyspark.mllib.clustering import KMeans
```







**Collaborative filtering**

- Collaborative ,ltering is ,nding users that share common interests
- Collaborative ,ltering is commonly used for recommender systems
- Collaborative ,ltering approaches
  - User-User Collaborative  ltering: Finds users that are similar to the target user
  - Item-Item Collaborative  ltering: Finds and recommends items that are similar to items with the target user



**Classification** is a supervised machine learning algorithm for sorting the input data into different categories





**Hadoop**

- volume
- velocity
- variety
- veracity





HDFS ：Hadoop Distributed File System

Hive is SQL engine on Hadoop





**ETL**

Extract, Transform, Load

- Data intergration
- Schema transformation : 3NF-> Star Schema
- Data standardization

ETL cycles:

- Extract raw data
- Cleaning, recode, derive data elements
- Load to target schema

Challenges

- Data Lineage
- Process rebustness
- Performance







A good ETL process is 4R

- **Readable** – process and code is well designed in the way that it is intuitive for people to understand.
- **Repeatable** – process should be deterministic. Given the same set of parameters, the process always produce the same results.
- **Recoverable** – process can recover from a set of unpredicted failures.
- **Resource-aware** – efficient process to utilize compute resources wisely



**Snoflake** : Cloud native Datawarehouse

- Scalable
- Software as service (zero installation)
- MPP Computing
- Split Storage and Computing
- Modernized SQL language : native support to json .
- Cloud native security integration
- Lots of other convenience features (DR, replication, data share, etc. etc. )



![image-20210528141736036](https://tva1.sinaimg.cn/large/008i3skNgy1gqyp2geun9j31jx0u07wi.jpg)















## AWS Cloud





[Leveraging AWS for Successful Data Engineering Strategy (onica.com)](https://onica.com/blog/data-analytics/aws-data-engineering-strategy/)





S3

EC2

Lambda



**AWS Lambda** is an on-demand cloud computing resource offered as function-as-a-service by AWS. 





**AWS Glue** is a fully managed ETL (extract, transform, and load) service that makes it simple and cost-effective to categorize your data, clean it, enrich it, and move it reliably between various data stores and data streams. 

AWS Glue is designed to work with semi-structured data.



You can use AWS Glue to organize, cleanse, validate, and format data for storage in a data warehouse or data lake. 

You can use AWS Glue when you run serverless queries against your Amazon S3 data lake. 

You can create event-driven ETL pipelines with AWS Glue. 

You can use AWS Glue to understand your data assets. 



Amazon Athena

Amazon SageMaker console 







Data pipeline

![image-20210530210948713](https://tva1.sinaimg.cn/large/008i3skNgy1gr1c7uue3dj30lm0okqdw.jpg)

1. Collecttion 

2. Extraction - Processing -Batch, Real-time

3. Transformation 

   1. Basic trandformations: affects  the apprearance and format of data
   2. Advanced trandformations:  severe content and relationships changing

4. Destinaiton -  data warehouse

   ![image-20210530211321532](https://tva1.sinaimg.cn/large/008i3skNgy1gr1cbi2j93j315q0de14i.jpg)

5. Monitoring

   Aws data pipeline

   ![image-20210530211500700](https://tva1.sinaimg.cn/large/008i3skNgy1gr1cd7ltjuj317o0imk98.jpg)







Data pipeline for machine  learning

![image-20210530211600231](https://tva1.sinaimg.cn/large/008i3skNgy1gr1ce95o9fj323s0iob03.jpg)

Tools to build machine learning data pipeline

ML-kit

Amazon SageMaker



Tools for general operations with data pipelines



ETL Data preparatioon and data intergeation:

Aws Glue, Apache spark

Data warehouse

Aws  redshift, snowflake

Batch Schedulers

Airflow, luigi

Stream processing

Spark, Flink, kafka, amazon kinesis





what is meant by a data pipeline well

it is a series of tools and actions for organizing and transferring the data to different storage and analysis systems

it automates the etl process extraction transformation load as a data pipeline







S3

S3 cli commands

1. Aws s3 mb
2. Aws s3 ls
3. Aws s3 rm
4. Aws s3 mv
5. Aws s3 cp

How do you protect data at rest in S3?

Entryption

1. AES-256
2. AWS-KMS



Explain when do you use ELB, ALB, NLB

- ALB : Layer 7, Path based routing, Attach WAF
- NLB : Layer 4 (Eg: Video streaming)
- CLB : Legacy, doesnt support TG



EBS vs EFS vs S3

EBS: Ablock storage, it is really fast, need an EC2 instance

EFS: managed service, accessed by multiple EC2 instance

S3: An obkect store, Great for log



Types of EC2 instances

R: memory optimized

C: Compute optimized

M: Medium

I: Storage Optimized

G: GPU

T: Burstable







Explain about any 5 Cloudformation functions?

- fn:: Join
- fn:: FindInMap
- fn:: Select
- fn:: Base64
- Ref












































# Summary



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







## Cloud

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

Apache hadoop

1. HDFS
2. MapReduce
3. Hive
   1. runs in hadoop
   2. structured query language: Hive SQL
   3. Initially Mapreduce, now other tools



Spark

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
- Josn : javascript object notation, semi-structured, atomic:number,string,bopolean, null, composite:array,object








































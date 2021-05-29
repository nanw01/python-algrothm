# Introduction to Airflow in Python

# Introduction to Airflow

## 1. Introduction to Airflow

Welcome to Introduction to Airflow! I'm Mike Metzger, a Data Engineer, and I'll be your instructor while we learn the components of Apache Airflow and why you'd want to use it. Let's get started!

## 2. What is data engineering?

Before getting into workflows and Airflow, let's discuss a bit about **data engineering**. While there are many specific definitions based on context, the general meaning behind data engineering is taking any action involving data and making it a reliable, repeatable, and maintainable process.

## 3. What is a workflow?

Before we can really discuss Airflow, we need to talk about workflows. A workflow is a set of steps to accomplish a given data engineering task. These can include any given task, such as downloading a file, copying data, filtering information, writing to a database, and so forth. A workflow is of varying levels of complexity. Some workflows may only have 2 or 3 steps, while others consist of hundreds of components. The complexity of a workflow is completely dependent on the needs of the user. We show an example of a possible workflow to the right. It's important to note that we're defining a workflow here in a general data engineering sense. This is an informal definition to introduce the concept. As you'll see later, workflow can have specific meaning within specific tools.

## 4. What is Airflow?

Airflow is a platform to program workflows (general), including the creation, scheduling, and monitoring of said workflows.

## 5. Airflow continued...

Airflow can use various tools and languages, but the actual workflow code is written with Python. **Airflow implements workflows as DAGs, or Directed Acyclic Graphs.** We'll discuss exactly what this means throughout this course, but for now think of it as a set of tasks and the dependencies between them. Airflow can be accessed and controlled **via code, via the command-line, or via a built-in web interface.** We'll look at all these options later on.

1. 1 https://airflow.apache.org/docs/stable/

## 6. Other workflow tools

Airflow is not the only tool available for running data engineering workflows. Some other options are **Spotify's Luigi, Microsoft's SSIS, or even just Bash scripting.** We'll use some Bash scripting within our Airflow usage, but otherwise we'll focus on Airflow.

## 7. Quick introduction to DAGs

A DAG stands for a Directed Acyclic Graph. In Airflow, this represents the set of tasks that make up your workflow. It consists of the tasks and the dependencies between tasks. DAGs are created with various details about the DAG, including the name, start date, owner, email alerting options, etc.

## 8. DAG code example

We will go into further detail in the next lesson but a very simple DAG is defined using the following code. A new DAG is created with the dag_id of etl_pipeline and a default_args dictionary containing a start_date for the DAG. Note that within any Python code, this is referred to via the variable identifier, etl_dag, but within the Airflow shell command, you must use the dag_id.

```python
etl_dag = DAG(
    dag_id='etl_pipeline',
    default_args={"start_date": "2020-01-08"}
)
```



## 9. Running a workflow in Airflow

To get started, let's look at how to run a component of an Airflow workflow. These components are called tasks and simply represent a portion of the workflow. We'll go into further detail in later chapters. There are several ways to run a task, but one of the simplest is using the airflow run shell command. Airflow run takes three arguments, a dag_id, a task_id, and a start_date. All of these arguments have specific meaning and will make more sense later in the course. For our example, we'll use a dag_id of example-etl, a task named download-file, and a start date of 2020-01-10. This task would simply download a specific file, perhaps a daily update from a remote source. Our command as such is airflow run example-etl download-file 2020-01-10. This will then run the specified task within Airflow.

```shell
airflow run <dag_id> <task_id> <start_date>

airflow run example-etl download-file 2020-01-10
```



## 10. Let's practice!

We've looked at Airflow and some of the basic aspects of why you'd use it. We've also looked at how to run a task within Airflow from the command-line. Let's practice what we've learned.







# Airflow DAGs

## 1. Airflow DAGs

Welcome back! You've successfully interacted with some basic Airflow workflows via the command line. Let's now take a look at the primary building block of those workflows - the DAG.

## 2. What is a DAG?

Our first question is what is a DAG? Beyond any specific mathematical meaning, a DAG, or **Directed Acyclic Graph**, has the following attributes: It is **Directed**, meaning there is an inherent flow representing the dependencies or order between execution of components. These dependencies (even implicit ones) provide context to the tools on how to order the running of components. A DAG is also **Acyclic** - it does not loop or repeat. This does not imply that the entire DAG cannot be rerun, only that the individual components are executed once per run. In this case, a **Graph** represents the components and the relationships (or dependencies) between them. The term DAG is found often in data engineering, not just in Airflow but also Apache Spark, Luigi, and others.

1. 1 https://en.m.wikipedia.org/wiki/Directed_acyclic_graph

## 3. DAG in Airflow

As we're working with Airflow, let's look at its implementation of the DAG concept. Within Airflow, DAGs are written in Python, but can use components written in other languages or technologies. This means we'll define the DAG using Python, but we could include Bash scripts, other executables, Spark jobs, and so on. **Airflow DAGs are made up of components to be executed, such as operators, sensors, etc. Airflow typically refers to these as tasks.** We'll cover these in much greater depth later on, but for now think of a task as a thing within the workflow that needs to be done. Airflow DAGs contain dependencies that are defined, either explicitly or implicitly. These dependencies define the execution order so Airflow knows which components should be run at what point within the workflow. For example, you would likely want to copy a file to a server prior to trying to import it to a database.

## 4. Define a DAG

Let's look at defining a simple DAG within Airflow. When defining the DAG in Python, you must **first import the DAG object from airflow dot models**. Once imported, we **create a default arguments dictionary consisting of attributes that will be applied to the components of our DAG.** These attributes are optional, but provide a lot of power to define the runtime behavior of Airflow. Here we define the owner name as jdoe, an email address for any alerting, and specify the start date of the DAG. The start date represents the earliest datetime that a DAG could be run. Finally, we define our DAG object with the first argument using a name for the DAG, etl underscore workflow, and assign the default arguments dictionary to the default underscore args argument. There are many other optional configurations we will use later on. Note that the entire DAG is assigned to a variable called etl underscore dag. This will be used later when defining the components of the DAG, but the variable name etl underscore dag does not actually appear in the Airflow interfaces. Note, DAG is case sensitive in Python code.

```python
from airflow.models import DAG
from datetime import datetime
default_arguments = {
    'owner': 'jdoe',
    'email': 'jdoe@datacamp.com',
    'start_date': datetime(2020, 1, 20)
}
etl_dag = DAG( 'etl_workflow', default_args=default_arguments )
```



## 5. DAGs on the command line

When working with DAGs (and Airflow in general), you'll often want to use the airflow command line tool. The airflow command line program contains many subcommands that handle various aspects of running Airflow. You've used a couple of these already in previous exercises. Use the airflow dash h command for help and descriptions of the subcommands. Many of these subcommands are related to DAGs. You can use the airflow list underscore dags option to see all recognized DAGs in an installation. When in doubt, try a few different commands to find the information you're looking for.

```shell
Using airflow :
The airflow command line program contains many subcommands.
airflow -h for descriptions.
Many are related to DAGs.
airflow list_dags to show all recognized DAGs.
```



## 6. Command line vs Python

You may be wondering when to use the Airflow command line tool vs writing Python. In general, the airflow command line program is used to start Airflow processes (ie, webserver or scheduler), manually run DAGs or tasks, and review logging information. Python code itself is usually used in the creation and editing of a DAG, not to mention the actual data processing code itself.

![image-20210324202525022](https://i.loli.net/2021/03/25/2BAzuZqtwQig34L.png)

## 7. Let's practice!

Now that we've covered the basics of a DAG in Airflow and how to create one, let's practice working with DAGs!



```python

# Import the DAG object
from airflow.models import DAG

# Define the default_args dictionary
default_args = {
  'owner': 'dsmith',
  'start_date': datetime(2020, 1, 14),
  'retries': 2
}

# Instantiate the DAG object
etl_dag = DAG('example_etl', default_args=default_args)

```









# Airflow web interface

**Got It!**

## 1. Airflow web interface

Hello again! Now that we've looked at some basic DAG functionality and interacting with them, it's time to introduce the Airflow web UI.

## 2. DAGs view

The Airflow web UI is made up of several primary page groups useful in developing and administering workflows on the Airflow platform. Note that for this course, we'll only be focusing on a few pages but it's helpful to click around the various options and get familiar with what's available. The DAGs view of the Airflow UI is the page we'll spend most of our time.

## 3. DAGs view DAGs

It provides a quick status of the number of DAGs / workflows available.

## 4. DAGs view schedule

It shows us the schedule for the DAG (in date or cron format).

## 5. DAGs view owner

We can see the owner of the DAG.

## 6. DAGs view recent tasks

which of the most recent tasks have run,

## 7. DAGs view last run

when the last run started,

## 8. DAGs view last three

and the last three DAG runs.

## 9. DAGs view links

The links area on the right gives us quick access to many of the DAG specific views.

## 10. DAGs view example_dag

Don't worry about those for now - instead we'll click on the "example_dag" link which takes us to our DAG detail page.

## 11. DAG detail view

The DAG detail view gives us specific access to information about the DAG itself, including several views of information (Graph, Tree, and Code) illustrating the tasks and dependencies in the code. We also get access to the Task duration, task tries, timings, a Gantt chart view, and specific details about the DAG. We have the ability to trigger the DAG (to start), refresh our view, and delete the DAG if we desire. The detail view defaults to the Tree view, showing the specific named tasks, which operators are in use, and any dependencies between tasks. The circles in front of the words represent the state of the task / DAG. In the case of our specific DAG, we see that we have one task called generate_random_number.

## 12. DAG graph view

The DAG graph view arranges the tasks and dependencies in a chart format - this provides another view into the flow of the DAG. You can see the operators in use and the state of the tasks at any point in time. The tree and graph view provide different information depending on what you'd like to know. Try moving between them when examining a DAG to obtain further details. For this view we again see that we have a task called generate_random_number. We can also see that it is of the type BashOperator in the middle left of the image.

## 13. DAG code view

The DAG code view does exactly as it sounds - it provides a copy of the Python code that makes up the DAG. The code view provides easy access to exactly what defines the DAG without clicking in various portions of the UI. As you use Airflow, you'll determine which tools work best for you. It is worth noting that the code view is read-only. Any DAG code changes must be done via the actual DAG script. In this view, we can finally see the code making up the generate_random_number task and that it runs the bash command echo $RANDOM.

## 14. Logs

The Logs page, under the Browse menu option, provides troubleshooting and audit ability while using Airflow. This includes items such as starting the Airflow webserver, viewing the graph or tree nodes, creating users, starting DAGs, etc. When using Airflow, look at the logs often to become more familiar with the types of information included, and also what happens behind the scenes of an Airflow install. Note that you'll often refer to the Event type present on the Logs view when searching (such as graph, tree, cli scheduler).

## 15. Web UI vs command line

In most circumstances, you can choose between using the Airflow web UI or the command line tool based on your preference. The web UI is often easier to use overall. The command line tool may be simpler to access depending on settings (via SSH, etc.)

## 16. Let's practice!

Now that we've covered some of the most important pages of the Airflow UI, let's practice examining some workflows using it.





```shell
airflow webserver -p 9090
```



cli_scheduler event represents the presence of the Airflow scheduler process, a default process for our installation.





# Implementing Airflow DAGs



# Airflow operators

**Got It!**

## 1. Airflow operators

Welcome back! We've discussed the basics of tasks in Airflow without covering exactly what a task consists of. The most common task in Airflow is the Operator. Let's take a look at Operators now.

## 2. Operators

Airflow operators represent a single task in a workflow. This can be any type of task from **running a command, sending an email, running a Python script**, and so on. Typically Airflow operators **run independently** - meaning that all resources needed to complete the task are contained within the operator. Generally, Airflow operators do not share information between each other. This is to simplify workflows and allow Airflow to run the tasks in the most efficient manner. It is possible to share information between operators, but the details of how are beyond this course. Airflow contains many various operators to perform different tasks. For example, the DummyOperator can be used to represent a task for troubleshooting or a task that has not yet been implemented. We are focusing on the BashOperator for this lesson but will look at the PythonOperator and several others later on.

## 3. BashOperator

**The BashOperator executes a given Bash command or script. This** command can be pretty much anything Bash is capable of that would make sense in a given workflow. The BashOperator **requires three arguments:** the task id which is the name that shows up in the UI, the bash command (the raw command or script), and the dag it belongs to. The BashOperator runs the command in a temporary directory that gets automatically cleaned up afterwards. It is possible to specify environment variables for the bash command to try to replicate running the task as you would on a local system. If you're unfamiliar with environment variables, these are run-time settings interpreted by the shell. It provides flexibility while running scripts in a generalized way. The first example runs a bash command to echo Example exclamation mark to standard out. The second example uses a predefined bash script for its command, runcleanup.sh.

```python
BashOperator(
    task_id='bash_example',
    bash_command='echo "Example!"',
    dag=ml_dag
)
BashOperator(
    task_id='bash_script_example',
    bash_command='rncleanup.sh',
    dag=ml_dag
)
```

## 4. BashOperator examples

Before using the BashOperator, it must be imported from airflow dot operators dot bash_operator. The first example creates a BashOperator that takes a task_id, runs the bash command "echo 1", and assigns the operator to the dag. Note that we've previously defined the dag in an earlier exercise. The second example is a BashOperator to run a quick data cleaning operation using cat and awk. Don't worry if you don't understand exactly what this is doing. This is a common scenario when running workflows - you may not know exactly what a command does, but you can still run it in a reliable way.

```python
from airflow.operators.bash_operator import BashOperator

example_task = BashOperator(
    task_id='bash_ex',
    bash_command='echo 1',
    dag=dag
)

bash_task = BashOperator(
    task_id='clean_addresses',
    bash_command='cat ddresses.txt | awk "NF==10" > cleaned.txt',
    dag=dag
)
```

## 5. Operator gotchas

There are some general gotchas when using Operators. **The biggest is that individual operators are not guaranteed to run in the same location or environment.** This means that just because one operator ran in a given directory with a certain setup, it does not necessarily mean that the next operator will have access to that same information. If this is required, you must explicitly set it up. You may need to set up environment variables, especially for the BashOperator. For example, it's common in bash to use the tilde character to represent a home directory. This is not defined by default in Airflow. Another example of an environment variable could be AWS credentials, database connectivity details, or other information specific to running a script. Finally, it can also be tricky to run tasks with any form of elevated privilege. This means that any access to resources must be setup for the specific user running the tasks. If you're uncertain what elevated privileges are, think of running a command as root or the administrator on a system.

## 6. Let's practice!

We've discussed the basics of Airflow operators - Let's practice using them in some workflows now.

```python
# Import the BashOperator
from airflow.operators.bash_operator import BashOperator

# Define the BashOperator 
cleanup = BashOperator(
    task_id='cleanup_task',
    # Define the bash_command
    bash_command='cleanup.sh',
    # Add the task to the dag
    dag=analytics_dag
)

```



```python
# Define a second operator to run the `consolidate_data.sh` script
consolidate = BashOperator(
    task_id='consolidate_task',
    bash_command='consolidate_data.sh',
    dag=analytics_dag)

# Define a final operator to execute the `push_data.sh` script
push_data = BashOperator(
    task_id='pushdata_task',
    bash_command='push_data.sh',
    dag=analytics_dag)

```





# Airflow tasks

**Got It!**

## 1. Airflow tasks

Welcome back! Now that you've worked with operators a bit, let's take a look at the concept of tasks within Airflow.

## 2. Tasks

Within Airflow, tasks are instantiated operators. It basically is a shortcut to refer to a given operator within a workflow. Tasks are usually assigned to a variable within Python code. Using a previous example, we assign the BashOperator to the variable example underscore task. Note that within the Airflow tools, this task is referred by its task id, not the variable name.

## 3. Task dependencies

**Task dependencies in Airflow define an order of task completion**. While not required, task dependencies are usually present. **If task dependencies are not defined, task execution is handled by Airflow itself with no guarantees of order.** Task dependencies are referred to as upstream or downstream tasks. An upstream task means that it must complete prior to any downstream tasks. Since Airflow 1.8, task dependencies are defined using the bitshift operators. The upstream operator is two greater-than symbols. The downstream operator is two less-than symbols.

![image-20210324210301353](https://i.loli.net/2021/03/25/dGS9VjkyCJu7WoR.png)

## 4. Upstream vs Downstream

It's easy to get confused on when to use an **upstream or downstream operator**. The simplest analogy is that upstream means before and downstream means after. This means that any upstream tasks would need to complete prior to any downstream ones.

## 5. Simple task dependency

Let's look at a simple example involving two bash operators. We define our first task, and assign it to the variable task1. We then create our second task and assign it to the variable task2. Once each operator is defined and assigned to a variable, we can define the task order using the bitshift operators. In this case, we want to run task1 before task2. The most readable method for this is using the upstream operator, two greater-than symbols, as task1 upstream operator task2. Note that you could also define this in reverse using the downstream operator to accomplish the same thing. In this case, it'd be task2 two less-than symbols task1.

```python
# Define the tasks
task1 = BashOperator(task_id='first_task',bash_command='echo 1',dag=example_dag)
task2 = BashOperator(task_id='second_task',bash_command='echo 2',dag=example_dag)
# Set first_task to run before second_task
task1 >> task2 # or task2 << task1
```

## 6. Task dependencies in the Airflow UI

Let's take a look at what the Airflow UI shows for tasks and their dependencies. In this case, we're looking at the graph view within the Airflow web interface.

## 7. Task dependencies in the Airflow UI

Note that in the task area, our two tasks, first_task and second_task, are both present, but there is no order to the task execution. This is the DAG prior to setting the task dependency using the bitshift operator.

## 8. Task dependencies in the Airflow UI

Now let's look again at the view with a defined order via the bitshift operators. The view is similar but we can see the order of tasks indicated by the directed arrow between first underscore task and second underscore task.

## 9. Multiple dependencies

Dependencies can be as complex as required to define the workflow to your needs. We can chain a dependency, in this case setting task1 upstream of task2 upstream of task3 upstream of task4. The Airflow graph view shows a dependency view indicating this order. You can also mix upstream and downstream bitshift operators in the same workflow. If we define task1 upstream of task2 then downstream of task3, we get a configuration different than what we might expect. This creates a DAG where first underscore task and third underscore task must finish prior to second underscore task. This means we could define the same dependency graph on two lines, in a possibly clearer form. task1 upstream of task2. task3 upstream of task2. Note that because we don't require it, either task1 or task3 could run first depending on Airflow's scheduling.

```python
Chained dependencies:
task1 >> task2 >> task3 >> task4

Mixed dependencies:
task1 >> task2 << task3
or:
task1 >> task2
task3 >> task2
```

![image-20210324210548224](https://i.loli.net/2021/03/25/4qFgJHcAlZBC7vt.png)

## 10. Let's practice!

There are many intricacies to defining tasks and using the bitshift operators. The best way to solidify these is practice in the exercises.





```python
# Define a new pull_sales task
pull_sales = BashOperator(
    task_id='pullsales_task',
    bash_command='wget https://salestracking/latestinfo?json',
    dag=analytics_dag
)

# Set pull_sales to run prior to cleanup
pull_sales >> cleanup

# Configure consolidate to run after cleanup
consolidate << cleanup

# Set push_data to run last
consolidate >> push_data
```



# Additional operators

**Got It!**

## 1. Additional operators

Welcome back! Now that we've used the BashOperator and worked with tasks, let's take a look at some more common operators available within Airflow.

## 2. PythonOperator

The PythonOperator is similar to the BashOperator, except that it runs a Python function or callable method. Much like the BashOperator, it requires a taskid, a dag entry, and most importantly a python underscore callable argument set to the name of the function in question. You can also pass arguments or keyword style arguments into the Python callable as needed. Our first example shows a simple printme function that writes a message to the task logs. We must first import the PythonOperator from the airflow dot operators dot python underscore operator library. Afterwards, we create our function printme, which will write a quick log message when run. Once defined, we create the PythonOperator instance called python underscore task and add the necessary arguments.

```python
from airflow.operators.python_operator import PythonOperator
def printme():
    print("This goes in the logs!")
python_task = PythonOperator(
    task_id='simple_print',
    python_callable=printme,
    dag=example_dag
)
```



## 3. Arguments

The PythonOperator supports adding arguments to a given task. This allows you to pass arguments that can then be passed to the Python function assigned to python callable. **The PythonOperator supports both positional and keyword style arguments as options to the task.** For this course, we'll focus on using keyword arguments only for the sake of clarity. To implement keyword arguments with the PythonOperator, we define an argument on the task called op underscore kwargs. This is a dictionary consisting of the named arguments for the intended Python function.

## 4. op_kwargs example

Let's create a new function called sleep, which takes a length of time argument. It uses this argument to call the time dot sleep method. Once defined, we create a new task called sleep underscore task, with the taskid, dag, and python callable arguments added as before. This time we'll add our op underscore kwargs dictionary with the length of time variable and the value of 5. Note that the dictionary key must match the name of the function argument. If the dictionary contains an unexpected key, it will be passed to the Python function and typically cause an unexpected keyword argument error.

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



## 5. EmailOperator

There are many other operators available within the Airflow ecosystem. The primary operators are in the airflow dot operators or airflow dot contrib dot operators libraries. Another useful operator is the EmailOperator, which as expected sends an email from within an Airflow task. It can contain the typical components of an email, including HTML content and attachments. Note that the Airflow system must be configured with the email server details to successfully send a message.

```python
from airflow.operators.email_operator import EmailOperator
email_task = EmailOperator(
    task_id='email_sales_report',
    to='sales_manager@example.com',
    subject='Automated Sales Report',
    html_content='Attached is the latest sales report',
    files='latest_sales.xlsx',
    dag=example_dag
)
```

## 6. EmailOperator example

A quick example for sending an email would be sending a generated sales report upon completion of a workflow. We first must import the EmailOperator object from airflow dot operators dot email underscore operator. We can then create our EmailOperator instance with the task id, the to, subject, and content fields and a list of any files to attach. Note that in this case we assume the file latest underscore sales dot xlsx was previously generated - later in the course we'll see how to verify that first. Finally we add it to our dag as usual.

## 7. Let's practice!

We've looked at a couple of new Airflow operators - let's practice using them in the exercises ahead.



```python
def pull_file(URL, savepath):
    r = requests.get(URL)
    with open(savepath, 'wb') as f:
        f.write(r.content)   
    # Use the print method for logging
    print(f"File pulled from {URL} and saved to {savepath}")

from airflow.operators.python_operator import PythonOperator

# Create the task
pull_file_task = PythonOperator(
    task_id='pull_file',
    # Add the callable
    python_callable=pull_file,
    # Define the arguments
    op_kwargs={'URL':'http://dataserver/sales.json', 'savepath':'latestsales.json'},
    dag=process_sales_dag
)
```



```python
# Add another Python task
parse_file_task = PythonOperator(
    task_id='parse_file',
    # Set the function to call
    python_callable=parse_file,
    # Add the arguments
    op_kwargs={'inputfile':'latestsales.json', 'outputfile':'parsedfile.json'},
    # Add the DAG
    dag=process_sales_dag
)
```





```python
# Import the Operator
from airflow.operators.email_operator import EmailOperator

# Define the task
email_manager_task = EmailOperator(
    task_id='email_manager',
    to='manager@datacamp.com',
    subject='Latest sales JSON',
    html_content='Attached is the latest sales JSON file as requested.',
    files='parsedfile.json',
    dag=process_sales_dag
)

# Set the order of tasks
pull_file_task >> parse_file_task >> email_manager_task
```



# Airflow scheduling

## 1. Airflow scheduling

Welcome back! We've spent most of this chapter implementing Airflow tasks within our workflows. Now let's take a look at what's required to schedule those workflows and have them run automatically.

## 2. DAG Runs

When referring to scheduling in Airflow, we must first discuss a DAG run. This is an instance of a workflow at a given point in time. For example, it could be the currently running instance, or it could be one run last Tuesday at 3pm. **A DAG can be run manually, or via the schedule interval parameter passed when the DAG is defined.** Each DAG run maintains a state for itself and the tasks within. **The DAGs can have a running, failed, or success state.** The individual tasks can have these states or others as well (ie, queued, skipped).

1. 1 https://airflow.apache.org/docs/stable/scheduler.html

## 3. DAG Runs view

Within the Airflow UI, you can view all DAG runs under the Browse: DAG Runs menu option. This provides the assorted details about any DAGs that have run within the current Airflow instance.

## 4. DAG Runs state

As mentioned, you can view the state of a DAG run within this page, illustrating whether the DAG run was successful or not.

## 5. Schedule details

When scheduling a DAG, there are many attributes to consider depending on your scheduling needs. The start date value specifies the first time the DAG could be scheduled. This is typically defined with a Python datetime object. The end date represents the last possible time to schedule the DAG. Max tries represents how many times to retry before fully failing the DAG run. The schedule interval represents how often to schedule the DAG for execution. There are many nuances to this which we'll cover in a moment.

![image-20210324212628714](https://i.loli.net/2021/03/25/3hZKqTUcxgylIMC.png)

## 6. Schedule interval

The schedule interval represents how often to schedule the DAG runs. The scheduling occurs between the start date and the potential end date. Note that this is not when the DAGs will absolutely run, but rather a minimum and maximum value of when they could be scheduled. The schedule interval can be defined by a couple methods - with a cron style syntax or via built-in presets.

![image-20210324212655772](https://i.loli.net/2021/03/25/f6BV1UJrSZOIcuh.png)

## 7. cron syntax

The cron syntax is the same as the format for scheduling jobs using the Unix cron tool. It consists of five fields separated by a space, starting with the minute value (0 through 59), the hour (0 through 23), the day of the month (1 through 31), the month (1 through 12), and the day of week (0 through 6). An asterisk in any of the fields represents running for every interval (for example, an asterisk in the minute field means run every minute) A list of values can be given on a field via comma separated values.

![image-20210324212646513](https://i.loli.net/2021/03/25/pjfc6sodJH9OhrB.png)

## 8. cron examples

The cron entry 0 12 asterisk asterisk asterisk means run daily at Noon (12:00) asterisk asterisk 25 2 asterisk represents running once per minute, but only on February 25th. 0 comma 15 comma 30 comma 45 asterisk asterisk asterisk asterisk means to run every 15 minutes.

## 9. Airflow scheduler presets

Airflow has several presets, or shortcut syntax options representing often used time intervals. The @hourly preset means run once an hour at the beginning of the hour. It's equivalent to 0 asterisk asterisk asterisk asterisk in cron. The @daily, @weekly, @monthly, and @yearly presets behave similarly.

1. 1 https://airflow.apache.org/docs/stable/scheduler.html

## 10. Special presets

Airflow also has two special presets for schedule intervals. None means don't ever schedule the DAG and is used for manually triggered workflows. @once means to only schedule a DAG once.

## 11. schedule_interval issues

Scheduling DAGs has an important nuance to consider. When scheduling DAG runs, Airflow will use the start date as the earliest possible value, but not actually schedule anything until at least one schedule interval has passed beyond the start date. Given a start_date of February 25, 2020 and a @daily schedule interval, Airflow would then use the date of February 26, 2020 for the first run of the DAG. This can be tricky to consider when adding new DAG schedules, especially if they have longer schedule intervals.

## 12. Let's practice!

We've covered a lot of ground in this lesson and chapter. Let's practice scheduling our workflows and we'll see each other back in chapter 3.



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



- Order the schedule intervals from least to greatest amount of time.

![image-20210324213257488](https://i.loli.net/2021/03/25/8NArVU9BzGlOstp.png)





# Maintaining and monitoring Airflow workflows

# Airflow sensors

**Got It!**

## 1. Airflow sensors

Welcome back! Now that we've gotten some practice using operators and tasks within Airflow, let's look at a special kind of operator called a sensor.

## 2. Sensors

A sensor is a special kind of operator that waits for a certain condition to be true. Some examples of conditions include waiting for the creation of a file, uploading a database record, or a specific response from a web request. With sensors, you can define how often to check for the condition(s) to be true. Since sensors are a type of operator, they are assigned to tasks just like normal operators. This means you can apply the bitshift dependencies to them as well.

## 3. Sensor details

All sensors are derived from the airflow dot sensors dot base underscore sensor underscore operator class. There are some default arguments available to all sensors, including mode, poke_interval, and timeout. The mode tells the sensor how to check for the condition and has two options, poke or reschedule. The default is poke, and means to continue checking until complete without giving up a worker slot. Reschedule means to give up the worker slot and wait for another slot to become available. We'll discuss worker slots in the next lesson, but for now consider a worker slot to be the capability to run a task. The poke_interval is used in the poke mode, and tells Airflow how often to check for the condition. This is should be at least 1 minute to keep from overloading the Airflow scheduler. The timeout field is how long to wait (in seconds) before marking the sensor task as failed. To avoid issues, make sure your timeout is significantly shorter than your schedule interval. Note that as sensors are operators, they also include normal operator attributes such as task_id and dag.

## 4. File sensor

A useful sensor is the FileSensor, found in the airflow dot contrib dot sensors library. The FileSensor checks for the existence of a file at a certain location in the file system. It can also check for any files within a given directory. A quick example is importing the FileSensor object, then defining a task called file underscore sensor underscore task. We set the task_id and dag entries as usual. The filepath argument is set to salesdata.csv, looking for a file with this filename to exist before continuing. We set the poke_interval to 300 seconds, or to repeat the check every 5 minutes until true. Finally, we use the bitshift operators to define the sensor's dependencies within our DAG. In this case, we must run init_sales_cleanup, then wait for the file_sensor_task to finish, then we run generate_report.

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

## 5. Other sensors

There are many types of sensors available within Airflow. The ExternalTaskSensor waits for a task in a separate DAG to complete. This allows a loose connection to other workflow tasks without making any one workflow too complex. The HttpSensor will request a web URL and allow you define the content to check for. The SqlSensor runs a SQL query to check for content. Many other sensors are available in the airflow dot sensors and airflow dot contrib dot sensors libraries.

## 6. Why sensors?

You may be wondering when to use a sensor vs an operator. For the most part, you'll want to use a normal operator unless you have any of the following requirements: You're uncertain when a condition will be true. If you know something will complete that day but it might vary by an hour or so, you can use a sensor to check until it is. If you want to continue to check for a condition but not necessarily fail the entire DAG immediately. This provides some flexibility in defining your DAG. Finally, if you want to repeatedly run a check without adding cycles to your DAG, sensors are a good choice.

## 7. Let's practice!

We've looked at a lot about sensors - let's practice using them now.



![image-20210324214959927](https://i.loli.net/2021/03/25/628iTFyjgrdMKI7.png)



# Airflow executors

## 1. Airflow executors

Now that we've implemented some Airflow sensors, let's discuss the execution model within Airflow.

## 2. What is an executor?

In Airflow, an executor is the component that actually runs the tasks defined within your workflows. Each executor has different capabilities and behaviors for running the set of tasks. Some may run a single task at a time on a local system, while others might split individual tasks among all the systems in a cluster. As mentioned in the previous lesson, this is often referred to as the number of worker slots available. We'll discuss some of these in more detail soon, but a few examples of executors are **the SequentialExecutor, the LocalExecutor, and the CeleryExecutor.** This is not an exhaustive list, and you can also create your own executor if required (though we won't cover that in this course).

## 3. SequentialExecutor

The SequentialExecutor is the **default execution engine** for Airflow. It runs **only a single task at a time**. This means having multiple workflows scheduled around the same timeframe may cause things to take longer than expected. The SequentialExecutor is **useful for debugging** as it's fairly simple to follow the flow of tasks and it can also be used with some integrated development environments (though we won't cover that here). The most important aspect of the SequentialExecutor is that while it's very **functional for learning and testing,** it's not **really recommended for production** due to the limitations of task resources.

## 4. LocalExecutor

The LocalExecutor is another option for Airflow that **runs entirely on a single system.** **It basically treats each task as a process on the local system, and is able to start as many concurrent tasks as desired / requested / and permitted by the system resources** (ie, CPU cores, memory, etc). This concurrency is the parallelism of the system, and it is defined by the user in one of two ways - either unlimited, or limited to a certain number of simultaneous tasks. Defined intelligently, the LocalExecutor is a good choice for a single production Airflow system and can utilize all the resources of a given host system.

## 5. CeleryExecutor

The last executor we'll look at is the **Celery executor.** If you're not familiar with Celery, it's a general queuing system written in Python that allows multiple systems to communicate as a basic cluster. Using a CeleryExecutor, multiple Airflow systems can be configured as workers for a given set of workflows / tasks. You can add extra systems at any time to better balance workflows. The power of the CeleryExecutor is significantly more difficult to setup and configure. It requires a working Celery configuration prior to configuring Airflow, not to mention some method to share DAGs between the systems (ie, a git server, Network File System, etc). While it is more difficult to configure, the CeleryExecutor is a powerful choice for anyone working with a large number of DAGs and / or expects their processing needs to grow.

## 6. Determine your executor

Sometimes when developing Airflow workflows, you may want to know the executor being used. If you have access to the command line, you can determine this by: Looking at the appropriate airflow dot cfg file. Search for the executor equal line, and it will specify the executor in use. Note that we haven't discussed the airflow.cfg file in depth as we assume a configured Airflow instance in this course. The airflow.cfg file is where most of the configuration and settings for the Airflow instance are defined, including the type of executor.

![image-20210324215417463](https://i.loli.net/2021/03/25/ucGAMi5DfXWCSb8.png)

## 7. Determine your executor #2

You can also determine the executor by running airflow list_dags from the command line. Within the first few lines, you should see an entry for which executor is in use (In this case, it's the SequentialExecutor).

![image-20210324215429911](https://i.loli.net/2021/03/25/5P2rtfMQFp6hegl.png)

## 8. Let's practice!

We've just discussed some of the various Airflow executors - let's practice what we've learned in the exercises ahead.



# Debugging and troubleshooting in Airflow

##  1. Debugging and troubleshooting in Airflow

Welcome back! Let's take a look at one of the biggest aspects of running a production system with Airflow and data engineering in general - debugging and troubleshooting.

## 2. Typical issues...

There are several common issues you may run across while working with Airflow - it helps to have an idea of what these might be and how best handle them. The first common issue is a DAG or DAGs that won't run on schedule. The next is a DAG that simply won't load into the system. The last common scenario involves syntax errors. Let's look at these more closely.

## 3. DAG won't run on schedule

The most common reason why a DAG won't run on schedule is the scheduler is not running. Airflow contains several components that accomplish various aspects of the system. The Airflow scheduler handles DAG run and task scheduling. If it is not running, no new tasks can run. You'll often see this error within the web UI if the scheduler component is not running. You can easily fix this issue by running airflow scheduler from the command-line.

## 4. DAG won't run on schedule

As we've covered before, another common issue with scheduling is the scenario where at least one schedule interval period has not passed since either the start date or the last DAG run. There isn't a specific fix for this, but you might want to modify the start date or schedule interval to meet your requirements. The last scheduling issue you'll often see is related to what we covered in the previous lesson - the executor does not have enough free slots to run tasks. There are basically three ways to alleviate this problem - by changing the executor type to something capable of more tasks (LocalExecutor or CeleryExecutor), by adding systems or system resources (RAM, CPUs), or finally by changing the scheduling of your DAGs.

## 5. DAG won't load

You'll often see an issue where a new DAG will not appear in your DAG view of the web UI or in the airflow list_dags output. The first thing to check is that the python file is in the expected DAGs folder or directory. You can determine the current DAGs folder setting by examining the airflow.cfg file. The line dags underscore folder will indicate where Airflow expects to find your Python DAG files. Note that the folder path must be an absolute path.

## 6. Syntax errors

Probably the most common reason a DAG workflow won't appear in your DAG list is one or more syntax errors in your python code. These are sometimes difficult to find, especially in an editor not setup for Python / Airflow (such as a base Vim install). I tend to prefer using Vim with some Python tools loaded, or VSCode but it's really up to your preference. There are two quick methods to check for these issues - airflow list_dags, and running your DAG script with python.

## 7. airflow list_dags

The first is to run airflow space list underscore dags. As we've seen before, Airflow will output some debugging information and the list of DAGs it's processed. If there are any errors, those will appear in the output, helping you to troubleshoot further.

## 8. Running the Python interpreter

Another method to verify Python syntax is to run the actual python3 interpreter against the file. You won't see any output normally as there's nothing for the interpreter to do, but it can check for any syntax errors in your code. If there are errors, you'll get an appropriate error message. If there are no errors, you'll be returned to the command prompt.

## 9. Let's practice!

Let's practice handling some of these common issues in the exercises ahead.





# SLAs and reporting in Airflow

##  1. SLAs and reporting in Airflow

Welcome back! Let's talk now about SLAs and reporting in Airflow.

## 2. SLAs

You may be wondering, what is an SLA? **SLA stands for Service Level Agreement.** Within the business world, this is often an uptime or availability guarantee. Airflow treats it a bit differently - it's considered the amount of time a task or a DAG should require to run. An SLA miss is any situation where a task or DAG does not meet the expected timing for the SLA. If an SLA is missed, an email alert is sent out per the system configuration and a note is made in the log. Any SLA miss can be viewed in the Browse, SLA Misses menu item of the web UI.

## 3. SLA Misses

To view any given SLA miss, you can access it in the web UI, via the Browse: SLA Misses link. It provides you general information about what task missed the SLA and when it failed. It also indicates if an email has been sent when the SLA failed.

## 4. Defining SLAs

There are several ways to define an SLA but we'll only look at two for this course. The first is via an sla argument on the task itself. This takes a timedelta object with the amount of time to pass. The second way is using the default_args dictionary and defining an sla key. The dictionary is then passed into the default_args argument of the DAG and applies to any tasks internally.

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

## 5. timedelta object

We haven't covered the timedelta object yet so let's look at some of the details. It's found in the datetime library, along with the datetime object. Most easily accessed with an import statement of from datetime import timedelta. Takes arguments of days, seconds, minutes, hours, and weeks. It also has milliseconds and microseconds available, but those wouldn't apply to Airflow. To create the object, you simply call timedelta with the argument or arguments you wish to reference. To create a 30 second time delta, call it with seconds equals 30. Or weeks equals 2. Or you can combine it into a longer mix of any of the arguments you wish (in this case, 4 days, 10 hours, 20 minutes, and 30 seconds).

```python
from datetime import timedelta

timedelta(seconds=30)
timedelta(weeks=2)
timedelta(days=4, hours=10, minutes=20, seconds=30)
```

## 6. General reporting

For reporting purposes you can use email alerting built into Airflow. There are a couple ways to do this. **Airflow has built-in options for sending messages on success, failure, or error / retry.** These are handled via keys in the default_args dictionary that gets passed on DAG creation. The required component is the list of emails assigned to the email key. Then there are boolean options for email underscore on underscore failure, email underscore on underscore retry, and email underscore on underscore success. In addition, we've already looked at the EmailOperator earlier but this is useful for sending emails outside of one of the defined Airflow options. Note that sending an email does require configuration within Airflow that is outside the scope of this course. The Airflow documentation provides information on how to set up the global email configuration.

```python
default_args={
  'email': ['airflowalerts@datacamp.com'],
  'email_on_failure': True,
  'email_on_retry': False,
  'email_on_success': True,
  ...
}
```



## 7. Let's practice!

Let's finish up this chapter by practicing what you've learned!



```python
# Import the timedelta object
from datetime import timedelta

# Create the dictionary entry
default_args = {
  'start_date': datetime(2020, 2, 20),
  'sla': timedelta(minutes=30)
}

# Add to the DAG
test_dag = DAG('test_workflow', default_args=default_args, schedule_interval='@None')
```



```python
# Define the email task
email_report = EmailOperator(
        task_id='email_report',
        to='airflow@datacamp.com',
        subject='Airflow Monthly Report',
        html_content="""Attached is your monthly workflow report - please refer to it for more detail""",
        files=['monthly_report.pdf'],
        dag=report_dag
)

# Set the email task to run after the report is generated
email_report << generate_report
```



```python
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime

default_args={
    'email': ['airflowalerts@datacamp.com','airflowadmin@datacamp.com'],
    'email_on_failure': True,
    'email_on_success':True
}
report_dag = DAG(
    dag_id = 'execute_report',
    schedule_interval = "0 0 * * *",
    default_args=default_args
)

precheck = FileSensor(
    task_id='check_for_datafile',
    filepath='salesdata_ready.csv',
    start_date=datetime(2020,2,20),
    mode='reschedule',
    dag=report_dag)

generate_report_task = BashOperator(
    task_id='generate_report',
    bash_command='generate_report.sh',
    start_date=datetime(2020,2,20),
    dag=report_dag
)

precheck >> generate_report_task

```





# Building production pipelines in Airflow

# Working with templates

## 1. Working with templates

Great work getting to the last chapter of this course! You've learned a lot, but let's take a look at some of the more advanced features of Airflow, starting with templates.

## 2. What are templates?

You may be wondering what templates are and what they do in the case of Airflow. Templates allow substitution of information during a DAG run. In other words, every time a DAG with templated information is executed, information is interpreted and included with the DAG run. Templates provide added flexibility when defining tasks. We'll see examples of this shortly. Templates are created using the Jinja templating language. A full explanation of Jinja is out of scope for this course, but we'll cover some basics in the coming slides.

## 3. Non-Templated BashOperator example

Before we get specifically into a templated example, let's consider what we would do for the following requirement. Your manager has asked you to simply echo the word "Reading" and a list of files to a log / output / etc. If we were to do this with what we currently know about Airflow, we would likely create multiple tasks using the BashOperator. Our first task would setup the task with the intended bash command - in this case echo Reading file1 dot txt, as an argument to the BashOperator. If we had a second file, we would create a second task using the bash command echo Reading file2 dot txt. This type of code would continue for the entire list of files. Consider what this would look like if we had 5, 10, or even 100+ files we needed to process. There would be a lot of repetitive code. Not to mention what if you needed to change the command being used / etc.

```python
 t1 = BashOperator(
       task_id='first_task',
       bash_command='echo "Reading file1.txt"',
       dag=dag)
t2 = BashOperator(
       task_id='second_task',
       bash_command='echo "Reading file2.txt"',
       dag=dag)
```



## 4. Templated BashOperator example

Let's take a look at how we would accomplish the same behavior with templates. First, we need to create a variable containing our template - which is really just a string with some specialized formatting. Our string is the actual bash command echo and instead of the file name, we're using two open curly braces, the term params dot filename, and then two closing curly braces. The curly braces when used in this manner represent information to be substituted. This will make more sense in a moment. If you've done any web development or worked with any reporting tools, you've likely worked with something similar. Next, we create our Airflow task as we have previously. We assign a task_id and dag argument as usual, but our bashcommand looks a little different. We set the bashcommand to use the templated command string we defined earlier. We also have an additional argument called params. In this case, params is a dictionary containing a single key filename with the value file1 dot txt. Now, if you look back at the templated command, you'll notice that the term in the curly braces is params.filename. At runtime, Airflow will execute the BashOperator by reading the templated command and replacing params dot filename with the value stored in the params dictionary for the filename key. In other words, it would pass the BashOperator echo Reading file1 dot txt. The actual log output would be Reading file1 dot txt (after the BashOperator executed the command).

```python
templated_command="""
  echo "Reading {{ params.filename }}"
"""
t1 = BashOperator(task_id='template_task',
       bash_command=templated_command,
       params={'filename': 'file1.txt'}
       dag=example_dag)
```



## 5. Templated BashOperator example (continued)

Now, let's consider one way to use templates for our earlier task of outputting Reading file1 dot txt and Reading file2 dot txt. First, we create our templated command as before. Next, create the first task and pass the params dict with a filename key and the value file1 dot txt. To pass another entry, we can create a second task and modify the params dict accordingly. This time the filename would contain file2 dot txt and Airflow would substitute that value instead. The resulting output would be as expected. Note, you may be wondering what templates do for you here. You'll see more in the coming exercises and lessons.



```python
templated_command="""
  echo "Reading {{ params.filename }}"
"""
t1 = BashOperator(task_id='template_task',
       bash_command=templated_command,
       params={'filename': 'file1.txt'}
       dag=example_dag)
t2 = BashOperator(task_id='template_task',
       bash_command=templated_command,
       params={'filename': 'file2.txt'}
       dag=example_dag)
```



## 6. Let's practice!

We'll have more to discuss about templates in the next lesson but let's practice what we've learned and I'll see you back shortly.





```python
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Create a templated command to execute
# 'bash cleandata.sh datestring'
templated_command="""
bash cleandata.sh {{ ds_nodash }}
"""

# Modify clean_task to use the templated command
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          dag=cleandata_dag)

```



```python
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Modify the templated command to handle a
# second argument called filename.
templated_command = """
  bash cleandata.sh {{ ds_nodash }} {{params.filename}}
"""

# Modify clean_task to pass the new argument
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          params={'filename': 'salesdata.txt'},
                          dag=cleandata_dag)

# Create a new BashOperator clean_task2
clean_task2 = BashOperator(task_id='cleandata_task2',
                          bash_command=templated_command,
                          params={'filename': 'supportdata.txt'},
                          dag=cleandata_dag)
                           
# Set the operator dependencies
clean_task >> clean_task2

```



# More templates

## 1. More templates

Welcome back! Hopefully you've cemented the basics of templates in your mind. Now let's look at some of the more powerful options they provide.

## 2. Quick task reminder

In our last lesson we were given the task of taking a list of filenames and printing "Reading filename" to the log or output. In our templated version, we created a templated command that substituted the filename value in place of params dot filename. We also used two tasks with different filename values to demonstrate one way to use templates without changing the actual bash command. Now, let's consider another way to perform the substitution.

```python
templated_command="""
echo "Reading {{ params.filename }}"
"""
t1 = BashOperator(task_id='template_task',
     bash_command=templated_command,
     params={'filename': 'file1.txt'}
     dag=example_dag)
```



## 3. More advanced template

Jinja templates can be considerably more powerful than we've used so far. It is possible to use a for construct to allow us to iterate over a list and output content accordingly. Let's change our templated command to the following. We start with an open curly brace and the percent symbol, then use a normal python command of for filename in params dot filenames then percent close brace. Then we modify our output line to be echo Reading open curly braces filename close curly braces. We then use a Jinja entry to represent the end of the for loop, open curly brace percent endfor percent close curly brace. Note that this is required to define the end of the loop, as opposed to Python's typical whitespace indention. Now let's look at our BashOperator task. It looks similar except we've modified the params key to be filenames, and the value is now a list with two strings, file1 dot txt and file2 dot txt. When Airflow executes the BashOperator, it will iterate over the entries in the filenames list and substitute them in accordingly. Our output is the same as before, with a single task instead of two. Consider too the difference in code if you had 100 files in the list?

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



## 4. Variables

As part of the templating system, Airflow provides a set of built-in runtime variables. These provide assorted information about DAG runs, individual tasks, and even the system configuration. Template examples include the execution date, which is ds in the double curly brace pairs. It returns the date in a 4 digit year dash 2 digit month dash 2 digit day format. There's also a ds underscore nodash variety to get the same info without dashes. Note that this is a string, not a python datetime object. Another variable available is the prev underscore ds, which gives the date of the previous DAG run in the same format as ds. The nodash variety is here as well. You can also access the full DAG object using the dag variable. Or you can use the conf object to access the current Airflow configuration within code. There are many more variables available - you can refer to the documentation for more information.

1. 1 https://airflow.apache.org/docs/stable/macros-ref.htm

```text
Execution Date: {{ ds }}
Execution Date, no dashes: {{ ds_nodash }}
Previous Execution date: {{ prev_ds }}
Prev Execution date, no dashes: {{ prev_ds_nodash }}  # YYYYMMDD
DAG object: {{ dag }}
Airflow config object: {{ conf }}
```



## 5. Macros

In addition to the other Airflow variables, there is also a macros variable. The macros package provides a reference to various useful objects or methods for Airflow templates. Some examples of these include the macros dot datetime, which is the Python datetime dot datetime object. The macros dot timedelta template references the timedelta object. A macros dot uuid is the same as Python's uuid object. Finally, there are also some added functions available, such as macros dot ds underscore add. It provides an easy way to perform date math within a template. It takes two arguments, a YYYYMMDD datestring and an integer representing the number of days to add (or subtract if the number is negative). Our example here would return April 20, 2020. These are not all the available macros objects - refer to the Airflow documentation for more info.

![image-20210325205424850](https://tva1.sinaimg.cn/large/008eGmZEly1gox0veoekrj32820i478m.jpg)

## 6. Let's practice!

We've seen several interesting aspects of Airflow templates - let's practice using them now.



```python
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

filelist = [f'file{x}.txt' for x in range(30)]

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Modify the template to handle multiple files in a 
# single run.
templated_command = """
  <% for filename in params.filenames %>
  bash cleandata.sh {{ ds_nodash }} {{ filename }};
  <% endfor %>
"""

# Modify clean_task to use the templated command
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          params={'filenames': filelist},
                          dag=cleandata_dag)

```



```python
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



# Branching

## 1. Branching

Nice work so far - we're nearing the end of our introduction to Airflow. Let's take a look at our last new concept, branching.

## 2. Branching

Branching provides the ability for conditional logic within Airflow. Basically, this means that tasks can be selectively executed or skipped depending on the result of an Operator. By default, we're using the **BranchPythonOperator**. There are other branching operators available and as with everything else in Airflow, you can write your own if needed. This is however outside the scope of this course. We **import the BranchPythonOperator from airflow dot operators dot** python underscore operator. The BranchPythonOperator works by running a function (the python underscore callable as with the normal PythonOperator). The function returns the name (or names) of the task_ids to run. This is best seen with an example.

```python
from airflow.operators.python_operator import BranchPythonOperator
```



## 3. Branching example

For our branching example, let's assume we've already defined our DAG and imported all the necessary libraries. Our first task is to create the function used with the python_callable for the BranchPythonOperator. You'll note that the asterisk asterisk kwargs argument is the only component passed in. Remember that this is a reference to a keyword dictionary passed into the function. In the function we first access the ds underscore nodash key from the kwargs dictionary. If you remember from our previous lesson, this is the template variable used to return the date in YYYYMMDD format. We take this value, convert it to an integer, and then run a check if modulus 2 equals 0. Basically, we're checking if a number is fully divisible by 2. If so, it's even, otherwise, it's odd. As such, we return either even underscore day underscore task, or odd underscore day underscore task.

```python
 def branch_test(**kwargs):
  if int(kwargs['ds_nodash']) % 2 == 0:
    return 'even_day_task'
  else:
    return 'odd_day_task'
```



## 4. Branching example

The next part of our code creates the BranchPythonOperator. This is like the normal PythonOperator, except we pass in the provide underscore context argument and set it to True. This is the component that tells Airflow to provide access to the runtime variables and macros to the function. This is what gets referenced via the kwargs dictionary object in the function definition. Now we don't show the code here, but let's assume we've created two tasks for even days, and two tasks for odd numbered days. We need to set the dependencies using the bitshift operators. First, we configure the dependency order for start task, branch task, then even day task and even day task2. Now we need to set the dependency order for the odd day tasks. As we've already defined the dependency for the start and branch tasks, we can set odd day task to follow the branch task, and the odd day task2 to follow that. You may be wondering why you'd set these dependencies if one set is not going to run. If you didn't set these dependencies, all the tasks would run as normal, regardless of what the branch operator returned.

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



## 5. Branching graph view

Let's look at the DAG in the graph view of the Airflow UI. You'll notice that we have a start task upstream of the branch task. The branch task then shows two paths, one to the odd day tasks, and the other to the even day tasks.

## 6. Branching even days

Let's look first at what happens if we run on an even numbered day. The start task executes as normal, then the branch task checks the ds_nodash value and determines this is an even day. It returns the value even underscore day underscore task, which is then executed by Airflow followed by the even day task2. Note that the odd day tasks are marked in light pink, which refers to them being skipped.

## 7. Branching odd days

For completeness, let's look at the output from a run on an odd day. The process is the same, except that the branch task selects odd day task instead and the even branch is marked skipped.

## 8. Let's practice!

We're nearly to the end of our course - let's practice working with branches now.

```python
# Create a function to determine if years are different
def year_check(**kwargs):
    current_year = int(kwargs['ds_nodash'][0:4])
    previous_year = int(kwargs['prev_ds_nodash'][0:4])
    if current_year == previous_year:
        return 'current_year_task'
    else:
        return 'new_year_task'

# Define the BranchPythonOperator
branch_task = BranchPythonOperator(task_id='branch_task', dag=branch_dag,
                                   python_callable=year_check, provide_context=True)
# Define the dependencies
branch_dag >> current_year_task
branch_dag >> new_year_task
```



# Creating a production pipeline

##  1. Creating a production pipeline

We're almost to the end of this course and we've covered an extensive amount about Airflow. Let's look at a few reminders before building out some production pipelines.

## 2. Running DAGs & Tasks

You may remember way back in chapter 1, we discussed how to run a task. If not, here's a quick reminder - use airflow run dag id task id and execution date from the command line. This will execute a specific DAG task as though it were running on the date specified. To run a full DAG, you can use the airflow trigger underscore dag dash e then the execution date and dag_id. This executes the full DAG as though it were running on the specified date.

```bash
 airflow run <dag_id> <task_id> <date>
  
 airflow trigger_dag -e <date> <dag_id>
```



## 3. Operators reminder

We've been working with operators and sensors through most of this course, but let's take a quick look at some of the most common ones we've used. The BashOperator behaves like most operators, but expects a bash underscore command parameter which is a string of the command to be run. The PythonOperator requires a python underscore callable argument with the name of the Python function to execute. The BranchPythonOperator is similar to the PythonOperator, but the python callable must be a function that accepts a kwargs entry. As such, the provide underscore context attribute must be set to true. The FileSensor requires a filepath argument of a string, and might need mode or poke underscore interval attributes. You can refer to previous chapters for further detail if required.

## 4. Template reminders

A quick reminder is that many objects in Airflow can use templates. Only certain fields can accept templated strings while others do not. It can be tricky to remember which ones support templates on what fields. One way to check is to use the built-in python documentation via a live python interpreter. To use this method, open a python3 interpreter at the command line. Import any necessary libraries (ie, the BashOperator) At the prompt, run help with the name of the Airflow object as the lone argument. Look for a line referencing template underscore fields. This line will specify if and which fields can use templated strings.

## 5. Template documentation example

This is an example of checking for help in the python interpreter. Notice the output with the template fields entry - in this case, the bash underscore command and the env fields can accept templated values.

## 6. Let's practice!

A final note before working through our last exercises - as a data engineer, your job is not to necessarily understand every component of a workflow. You may not fully understand all of a machine learning process, or perhaps how an Apache Spark job works. Your task is to implement any of those tasks in a repeatable and reliable fashion. Let's practice implementing workflows for the last time in this course now.





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





# Congratulations

## 1. Congratulations!

Congratulations on successfully completing this introduction to Airflow. We've covered a lot of ground since the first chapter and you should be pleased to have come this far. Let's take a moment to review what we've learned and cover some next steps.

## 2. What we've learned

Let's review everything we've worked with during this course. We started with learning about workflows and DAGs in Airflow. We've learned what an operator is and how to use several of the available ones. We learned about tasks and how they are defined by various types of operators. In addition, we learned about dependencies between tasks and how to set them with bitshift operators. We've used sensors to react to workflow conditions and state. We've scheduled DAGs in various ways. We used SLAs and alerting to maintain visibility on our workflows. We learned about the power of templating in our workflows for maximum flexibility when defining tasks. We've learned how to use branching to add conditional logic to our DAGs. Finally, we've learned about the Airflow interfaces (command line and UI), about Airflow executors, and a bit about how to debug and troubleshoot various issues with Airflow and our own workflows.

## 3. Next steps

A few suggestions for next steps include setting up your own environment for practice. You can follow the installation instructions in the Airflow documentation or use a cloud-based Airflow service. Look into other operators or sensors - there are operators available for Amazon's S3, Postgresql operators, HDFS sensors, and so forth. Experiment with dependencies with a large number of tasks. Consider how you expect the workflow to progress and always try to leave as much up to the scheduler as possible to achieve the best performance. Given the length of the course, there is only so much we could cover and we left out some important parts of Airflow such as XCom, connections, and managing the UI further. Refer to the documentation for more ideas. Finally and most importantly, keep building workflows. When you're uncertain how something works, try to build an example that covers what you'd like to accomplish. Look at the views within the Airflow UI to better understand how the system interprets your DAG code. The more you experiment, the better your understanding will grow.

## 4. Thank you!

Finally, thank you for taking this course and giving me the opportunity to introduce you to Airflow. Good luck on your future learning opportunities!














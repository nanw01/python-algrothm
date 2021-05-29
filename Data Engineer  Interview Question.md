



Data Engineer Interview Questions





what is Data Engineering?

Data engineering is a software engineering approach to developing and designing information  systems. It focuses on the collection and analysis of data. Conversion  of raw data into useful information



[Hadoop is an open-source software collection](https://www.upgrad.com/blog/what-is-hadoop-introduction-to-hadoop/?utm_source=MEDIUM&utm_medium=BODY&utm_campaign=MEDIUM_75472) of utilities that allow you to use a network of multiple computers for solving problems related to big data. I

Data Modeling



The Hadoop Distributed File System lets you store data in readily accessible forms. It saves your data in multiple nodes, which means it distributes the data. 

YARN is the acronym for ‘Yet Another Resource Negotiator’. It is a significant operation system and finds applications in Big Data processes.

[MapReduce ](https://www.webopedia.com/TERM/H/hadoop_mapreduce.html)is another powerful tool present in the Apache Hadoop collection. Its main job is to identify data and convert it into a suitable format for data processing.

Hadoop Common is a collection of free tools and software for Hadoop users. It’s a library of incredible tools that can make your job easier and more efficient.



A NameNode is a part of data storage in HDFS and tracks the different files present in clusters. NameNodes don’t store data. They store metadata of DataNodes, where HDFS stores its actual data.



In **[Hadoop](http://data-flair.training/blogs/hadoop-introduction-tutorial-quick-guide/)**, HDFS splits huge file into small chunks that is called **[Blocks](http://data-flair.training/blogs/data-blocks-hdfs-hadoop-distributed-file-system/)**. These are the smallest unit of data in file system.

**NameNode** (Master) will decide where data store in the**DataNode** (Slaves). All block of the files is the same size except the last block.



Velocity, Variety, Volume, and Veracity.



**COSHH** stands for Classification and Optimization-based Schedule for Heterogeneous Hadoop systems.



Star schema has a structure similar to a star; that’s why it has its name. The center of the star could have a fact table with various dimension tables associated with it. Data engineers use it to query substantial data sets.



A snowflake schema is a form of Star schema. The only difference is, it has additional dimensions, and it derives its name from its snowflake-like structure. It has normalized dimension tables, due to which it has other tables.



There are several core methods in Reducer. The first one is setup () that configures parameters, cleanup () cleans temporary data sets, and the Reducer runs reduce () method with every reduced task.



**FSCK** stands for File System Check. It’s a command of HDFS, and it uses this command to detect problems and inconsistencies in a file.



- Standalone mode
- Fully distributed mode
- Pseudo distributed mode.



YARN stands for Yet Another Resource Negotiator.



In Star schema, you have a higher chance of data redundancy, which is not the case with Snowflake schema. The DB design of Star schema is more straightforward than Snowflake. The complex join of Snowflake schema slows down its cube processing, which doesn't happen with Star schema.



In Hadoop, there are two kinds of nodes, NameNode and DataNode. The NameNode has the responsibility of storing the metadata of DataNodes and keep track of their status. DataNodes send signals to the NameNode to inform them that they are alive and are working. This signal is the **Heartbeat**.



Big Data

When you have humongous quantities of unstructured and structured data that you can’t process with conventional methods, it’s called big data. Big data is the field of analyzing and using highly complex data sets for gathering information. Traditional methods of data analysis don’t work well with such high quantities of complex data. In big data, data engineers have the task of analyzing raw data and convert it into usable data.



DAS stands for Direct Attached Storage, and NAS stands for Network Attached Storage. The storage capacity of NAS is 10⁹ to 10¹² in the byte. On the other hand, DAS has a storage capacity of 10⁹ bytes. The management costs of NAS are way less than DAS too.



In Hadoop, the distance between two nodes is equal to the sum of the length to their closest nodes. You can use getDistance() to find the distance between two nodes in Hadoop.



what are the design schemas thay are used when performing data modeling?



what are the differences between structured and unstructured data?







hadoop streaming


































# database design

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



**Data warehouses**

- Optimized for analytics - OLAP
  - Organized for reading/aggregating data
  - Usually read-only
- Contains data from multiple sources
- Massively Parallel Processing (MPP)
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



![image-20210523145033875](https://tva1.sinaimg.cn/large/008i3skNgy1gqt7fhvbegj314s0la0ui.jpg)



Data modeling : process of creating a data model for the data to be stored





Data Models

[Introduction to Data Modelling. What is Data Modelling? | by Sagar Lad | Sagar Explains Azure and Analytics : Data Engineering Series | Medium](https://medium.com/sagar-explains-azure-and-analytics-data-engineerin/introduction-to-data-modelling-c0c44432ec0b#:~:text=A data model helps design,to create a physical database.)

1. Conceptual data model

   describes entities, relationship, and attributes

2. Logical data model

   defines tables, cilumns, relationships

3. Physical data model

   Describe physical storage



Dimensional modeling

*Dimensional Modeling* (DM) is a data structure technique optimized for data storage in a Data warehouse.



Star schema

Dimensional  modeling: star schema

Fack tables

- Holds records of a metric
- Changes regularly
- Connects to dimensions via foreign keys

Demension tables

- Holds descriptioons of attributes
- Does not change as often

![image-20210523153438798](https://tva1.sinaimg.cn/large/008i3skNgy1gqt7ffszrcj31820j4jtm.jpg)

![image-20210523153506949](https://tva1.sinaimg.cn/large/008i3skNgy1gqt7fiykouj311a0j4tb3.jpg)



Same fact table, different dimensions

Star schemas : one dimension

Snowflake schemas : more than one dimension. Because dimension tables are normalized.

Normalization

Normal forms(NF)

Data anomalies : if **not** normalize enough

1. Updata anomaly
2. Insertion anomaly
3. Deletion anomaly



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







![image-20210523200040641](https://tva1.sinaimg.cn/large/008i3skNgy1gqt7fgufgrj30yq0jy403.jpg)

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




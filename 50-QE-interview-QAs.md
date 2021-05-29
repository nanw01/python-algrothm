##### what is Data Engineering?



##### Define Data Modeling



data modeling is a very

simple step of simplifying an entity

here in the concept of data engineering

you will be simplifying a complex

software by simply breaking it up into

diagrams 



##### what is the difference between structured and unstructured data



##### What are some of the import conponenets of hadoop?



Hadoop Common

HDFS

hadoop YARN

Hadoop Mapreduce



##### What is a namenode in hdfs?

one of Vital parts, as a way to store all hdfs data, to keep track of the files in all, data is actually storeed in the DataNodes and not in the NameNode.

##### What is hadoop Streaming?

Easily create maps and perform reduction operations. Later,n this can be submitteeed into a specific cluster for usage.



##### What are some of the important features of hadoop?

Open-sorce framework, works on the basis of distributed computing, parallel computing so fast data processing, data is stored in separate clusters away from the operations. data redundancy is given priority to ensur no data loss.



##### What is four V's of Big Data?

Volume, Variety,Veracity,Velocity

#####  What is block and block scanner in HDFS?

Block is considereed as a sigulaar entity of data, which is th e smallest factor.

Block is put into place to verify whether th eloss of blocks created by hadoop is put on the datanode successfully or not.

##### how does a block scanner handle corrupted files?



##### How does the namenode communicate with datanode?

Via messages. two messages, block reports and heartbeats.



##### What is meant by COSHH?

Classification and Optimazation Scheeduling for Hererogeneous Hadoop systems.

provides scheduling at both the cluster and hte application levels to directly have a positive impact on the completion time for jobs.







[Music]

hi guys and welcome to this session by

intel apat so we live in a data-driven

world there's huge chunks of information

everywhere and as a result there's a

huge demand for data engineers and so

today we have come up with this session

on interview questions for data

engineering so that you can be prepared

for your next interview but before we

get into all of that please subscribe to

our channel to never miss an update from

Intel about now without wasting any

further time let's just get into the

session well we already know that the

data engineers role is one of the top

roles to have in this particular decade

with that comes a lot of competition and

it where it creates a demand where

people are looking for the most

proficient data engineers they can find

and this ensures that there are a lot of

job openings which again equate to more

interviews and more chances of you guys

landing the dream job as data engineers

well without further ado let's begin

with this compilation of the top 50

questions that have the highest

probability of occurrence in an

interview and you can use this to ace

all of your interviews the questions and

answers of both precise and at the same

time descriptive enough to help you to

add on your own points to the answers as

well as answer them to the best of your

abilities so with this guy to be sure

that you can answer all of the high

probability occurrence questions in the

interview and eventually impress the

interviewer and land those jobs well let

us begin question number one what is

data engineering well data engineering

is a method it's a technique and in fact

it's a job role where a person is

proficient enough to handle the data and

work with data in terms of storing it

and converting a raw data into useful

information so data engineer is a person

who has the ability to or maintain the

data in the system work with the data

and eventually store it after the data

processing has been done so coming to

the next question what is data modeling

well

data modeling is a very

simple step of simplifying an entity

here in the concept of data engineering

you will be simplifying a complex

software by simply breaking it up into

diagrams when you think about diagrams

in data modelling think about flow

charts because flowcharts are a simple

representation of how a complex entity

can be broken down into a simple diagram

so this will basically give you a visual

representation and easier understanding

of the complex problem and even better

readability to a person who might not be

proficient in that particular software

usage as well coming to the next

question what are the design schemas

that are used when performing data

modeling well mainly there are two

schemas and you must understand when

you're learning about data engineering

it's the star schema and the snowflake

schema with the star schema basically

data is spread out in the structure of a

star where you have one primary table in

the middle and all of the other tables

are surrounding it are trickled down

from the main table when you have to

talk about the snowflake schema the

snowflake schema will consider a main

fact table and multiple dimensional

tables which will again have sub tables

as you can see on the right-hand side of

your screen so you have one fact table

you're a couple of dimensional tables

and these individual dimension tables

can again have n number of sub tables as

well so looking at it in a perspective a

star schema is very simple to implement

but then a snowflake schema is more

efficient and it will have a better

amount of usage in terms of efficiency

and responsiveness which we will be

checking out in the next couple of

questions this brings us to the fourth

question what are the differences

between structured and unstructured data

well we can compare four very important

parameters when we are answering this

question so whenever you're talking

about structured data make sure you

visualize an Excel workbook because this

will give you an understanding that your

data is sorted into rows and columns it

has sorted based on some factors so that

makes the primary difference between a

structured piece of data and an

unstructured piece of data a quick

example can be a data set as I mentioned

for the structured data and it can be

any sort of images we

or you know unorganized data in a text

file which can be unstructured as well

when we're talking about how these data

is stored coming to the first point

storage methodologies will be using a

database management system to analyze

maintain and work with a database to

work with a structured database when it

is unstructured data most of the storage

methodologies will depend on that

particular application and it goes

unmanaged in many of the cases as well

there are a couple of protocol standards

that you must know we have ODBC we have

the SQL and ad or dotnet for structured

data and we have the XML CSV SMTP and SM

SM our standards for the unstructured

data handling aspect as well when you're

talking about scaling this schemas be

the star schema or even the snowflake

schema in terms of structured data

making your schema into a more expansive

state or you know expanding on the

existing schema will be very difficult

in terms of structured data but then if

you have unstructured data you can

convert them into schemas very

efficiently and quickly and work with it

easily as well so these form some of the

vital difference is between structured

and unstructured data coming to the next

question

Hadoop is a very important part of a

data engineers role so understanding all

of the tools that come along with Hadoop

and the Apache ecosystem will add a lot

of weight H to your candidature so it is

fitting that question number five is

what is Hadoop well Hadoop is one of the

world's most used framework today for

handling big data hence it is called as

the gold standard of big data it is

basically an open source framework that

is used to perform all sorts of

activities such as data manipulation

data storage it has its own file storage

methodologies and it runs on entities

called as clusters will be checking out

a bit about this in the next set of

questions as well with that not moving

on to the next question question number

six what are the important components of

Hadoop well there are multiple

components that you can code at this

point of time but the four most

important concepts that you have to tell

out is these the first component is

Hadoop common well

common is this very important component

of Hadoop which basically consists of

all of the utilities all of the tools

all of the libraries and sub frameworks

that you'll be using in the hadoop

application this is very important and

the second thing is Hadoop file system

or as it's called HDFS so what HDFS does

is it basically provides a distributed

file system which will give the

application and the server client

architecture a high amount of bandwidth

where you know with respect to Big Data

you can transfer a lot of data at any

given moment and HDFS provides this

advantage to Hadoop users so coming to

Hadoop yarn yarn stands for yet another

resource negotiator and it is basically

a very beautifully put together tool

which is used to manage all of the

resources whenever you work with a

hadoop application so basically a yarn

provides a resource negotiator which in

turn is a simple scheduler so these set

of scheduling operation will add to the

advantages and the efficiency of the

system as well and the fourth and the

last important component is the Hadoop

MapReduce MapReduce is a technique which

is are being implemented everywhere ever

since the launch of a loop so with

MapReduce users can have access to a

large scale of data and process it very

effectively by making use of these two

functions one being the map function the

other one being the reduction function

so these form to be the very important

components of a loop of course you can

add on more components in case if the

interviewer is expecting you to know

more

in this particular case make sure that

you can now emphasize on the other

components which Hadoop provides as well

this brings us to the next question what

is a name node in HDFS well do

understand this name Lourdes is one of

the very very important aspects of HDFS

which is mostly asked in all of the

interviews because name node as an

entity in Hadoop which consists of all

of the metadata of the actual storage

files what you need to understand is

that when you're talking about name node

the actual data gets stored in the data

nodes there is another entity called as

data node and the data gets stored there

but their reference to those they

notes and all of the metadata which

describes the actual data is present in

the name node and this brings us to the

next question which is what is Hadoop

streaming well Hadoop streaming is a

very beautifully put together utility

which is provided by Hadoop to the users

in case if there is a requirement for

the users to go on to map operations and

work with reduction operations at the

same time as well so basically by

performing mapping operations and

reduction operations the user the user

has full ability and flexibility to work

with the data and process it in a

simplified manner and this is basically

used later to submit it to one cluster

where you know it can be put to use

after processing so so this is the state

in between where the data is converted

from its raw entity into information and

later where data is actually put into

use to drive some sort of analytics

question number nine what are some of

the important features of Hadoop well

the first most feature you need to talk

about is how it is open source since it

is open source it has a community of

millions of people across the globe or

where there are a lot of contributions

lot of revisions making the tool more

effective and easy to use with every

iteration and release and Hadoop works

on the basis of distributed computing

parallel processing and you know you

have to talk about data redundancy a bit

so since Hadoop works on the basis of

distributed computing your data is

spread across multiple machines and what

if our machine fails is your data lost

well the answer is no with respect to

Hadoop because data redundancy is

actually given priority here to make

sure that all of your data has backups

and there is a way to retrieve data in

case if it is lost so this is done by

storing your data in separate clusters

and actually performing various

operations which we will be checking out

in the next set of question this brings

us to question number 10 question number

10 seems to be one of the most common

and the highest probable occurrence of

fork for any question for a data

engineer's interview so it sets what are

the four V's of Big Data the four V's of

big data involve volume veracity

velocity

and variety well this is very important

that you guys know this volume is

talking about the amount of data that

you're supposed to handle whereas city

is talking about the quality of the data

how much of it is useful information how

much of it is just noise and then when

you're talking about velocity velocity

is all about the speed of arrival at

which you know the data is coming

through to the pipeline and this will

give you a clear idea of how quickly you

need to process the data and of course

variety as the name suggests is how

different your data from each other so

basically what are the types of data

that you are seeing that you need to

process your so with volume we're

talking about the amount of data where a

city talks about the quality of the data

velocity talks about how quickly the

data is arriving into your architecture

environment and a variety is talking

about the various types of data that can

be coming through a various sources as

well this brings us to the next question

question number 11 what is block and

block scanner in HDFS well whenever you

talk about a block make sure to

highlight on the aspect that it is a

singular entity of data this means that

a block is the smallest entity of data

when you're talking about Hadoop and

storing files in Hadoop because Hadoop

does this really nice thing or whenever

it encounters a large piece of fire or a

huge chunk of file it will automatically

cut it and slice it into tinier aspects

and these tinier aspects are the one we

call blocks and blocks scanner as the

name suggests a block scanner is

actually used to see if the large amount

of data which is cut into blocks is

actually correct or not so it checks if

there is any loss of blocks when Hadoop

goes on - or you know split the data

into multiple tiny entities and this

data as you might have already guessed

is getting stored on the data node and

all the information with respect to the

data gets stored in the name node as we

previously discussed so all in all the

block scanner is used to just verify if

if there is any loss of any number of

blocks whenever Hadoop goes on to split

the files question number 12 if there

are files which have been corrupted how

does a Hadoop and elza's or how does a

block scanner handle corrupted files so

you need to understand that whenever

there is you know whenever data is

stored in the data node there are

chances that a file might be corrupted

so the first couple of steps that happen

whenever Hadoop realizes that a through

block scanner that it has seen a

corrupted file is that the data node

will actually report this corruption

entry into the name node it the data

node literally tells the name nor that I

have a file which is corrupted and then

what the name though does is basically

it creates a replica of the file which

was corrupted and then this will give

you the original file which was

unaltered which is not corrupted and

then basically the name node tells the

data node back that the files have been

recreated and no need to worry and the

name node eventually tells the data node

again after the recreation of the

original files that these files have

been created so whenever there is a

match with respect to the replicas

created and the actual existing files

you need to understand that the data

block which was corrupted is actually

not removed so this is one important

point you can always mention coming to

question number 13 so how does the data

node communicate with the name node or

vice versa there are two important to

waste the name node and the data node

communicate with each other and the

concept is called as messages because

messages are sent in between name node

and data node and there are two types of

messages called as block reports and

heartbeats which are actually exchanged

and this forms the widest channel of

communication in between the data node

and the name node in Hadoop more on this

in the coming set of questions so

question number 14 what is meant by gosh

or Co SH H well this is a very simple

abbreviation for classification and

optimization based scheduling for

heterogeneous Hadoop systems so what it

basically means is that it provides very

good amount of tools and methodologies

required for scheduling activities which

are performed at the cluster level so

this will basically ensure that all of

your applications are completed on time

now that they have scheduling attached

to them and it is not happening in a

haphazard

so this is the basic working of Kosh

coming to question number 15 again since

we've already checked out schemas this

might be a follow-up question as well so

what is star schema in brief star schema

first of all is also called as the star

join schema make sure you remember that

and one of the most important things is

that star schema is the simpler schema

when you're comparing star and just a

snowflake schema as well so in the

concept of data warehousing it is a very

simple aspect of how data can be stored

and the structure as the name suggests

resembles a star and hence it's called

as a star schema your fact tables in the

middle and all the dimension tables

surrounding it so whenever we talk about

star schema understand that it will only

be used when you're working with huge

amounts of data of course you can still

use star schema when you're working with

a small amount of data but then it can

only be put to use effectively when

you're working with large amounts of

data so this question can again be asked

what does snowflake schema and brief

which can be which could be your

follower to the previous question as

well so when you're talking about

snowflake schema snowflake schema is

basically star schema version 2.0 think

of it that way because it is an

extension of star schema which has the

ability to have more dimensions attached

to it so basically it is a more complex

version of the star schema and it looks

like the structure of the snow snowflake

when looked at under a microscope hence

it has the name snowflake schema the

data in the snowflake schema is very

structured and it is split into many

more tables when you compare it to a

star schema which is performed basically

after normalizations this is done to

make sure your data is highly readable

and it is effectively stored as well and

then even after this if the interviewer

wants to push you a little bit more on

the basic schemas star and snowflake

here is a table which covers the

difference between the star schema and

the snowflake schema so these four

points are very important because in the

star schema you'll be emphasizing on the

aspect of where dimensional hierarchy is

stored and again in the case of

snowflake schema that it has stored and

every individual table out there

a hierarchy stored in separate tables in

the case of snowflake schema but in the

case of star schema it is stored in the

dimensional tables itself star schema

involves a lot of data redundancy while

there is low data redundancy in terms of

snowflake schemas and database designing

is very simple when you're working with

star schemas but it is very complex to

handle a lot of data in terms of store

storing it and maintaining it when

you're working with large-scale

snowflake schemas and of course this

ensures that star schema has faster

processing while snowflake schema in

some cases due to its nature might be a

little slower in data processing

activities this brings us to the 18th

question which says name the XML

configuration files which are present in

Hadoop do note that you do not have to

explain on these XML configuration files

but then make sure that he named the

files naming the files for this question

is very important and there are four

main XML configurations files present in

Hadoop their core site map rate site

which basically stands for MapReduce

then we have the HDFS site and the yard

site so make sure you name these four

configuration files when asked coming to

question number 19 question number 19

are was again asking you about another

very important and interesting aspect of

data engineering so the question goes

what is the meaning of fsck fsck

basically stands for filesystem check

make sure you remember this filesystem

check is nothing but fsck

and this is again a very important

command which is used to work with when

you're using the Hadoop file system so

whenever you go on to use the file check

basically it is performing a check where

you are analyzing your data to see if

there is any problem in the data or see

if any files are corrupted or if you

have to change anything in the data as

well so this will give the user a

first-hand look into the data to see if

there is anything wrong with it and make

sure to emphasize on the fact that it is

very important to perform file checks

especially in the world of data

engineering this brings us to question

number 20 question number 20 States

what

some of the methods of reducer well

whenever you talking about the map

function all the reducer function you

need to understand that each of these

entities will have sub methods involved

with them and when we talking about the

reducer here are the three methods which

will actually which are actually

associated with reducer the first thing

is set up the set up method is basically

used to understand what the input data

parameters are and also understand how

the data is cased and the protocols

which go on to gates the data as well

the second method involves clean up

clean up as the name suggests is simply

used to remove all of the cage files

after its usage or in the general

removal of any temporary files as well

and then when we're talking about the

radius method so the reduced method is

where the actual reduction operation

happens so it is called one time for

every key call and this basically forms

to be one of the most important method

in the entire aspect and working of

reducer as the name suggests moving on

to the next question what are the

different usage modes that that Hadoop

supports well well Hadoop basically is

used in three different modes the first

mode is standalone mode in standalone

mode the configuration files and the

data can be stored on the local machine

of the user itself you do not need to

have any sort of distributed

architecture client-server architecture

or whatever it is so basically all the

data is stored on the local machine then

when we talk about pseudo distributed

mode pseudo distributed mode basically

is the method of working where the

configuration files have to be present

in the local machine but the data can be

spread across a distributed system in a

fully distributed system of course the

configuration as well as the data can be

in the distributed environment overall

as well

of course you do not have to explain on

these three modes but it is always

advantageous to tell a little bit of all

these modes which will basically

strengthen are the aspects of the

interviewer where he understands that

you have put in some work and you know

these modes in detail this brings us to

question number 22 well question number

22 basically is this how is data

security and short in Hadoop

whenever you're working with large

amounts of data as a data engineer you

need to understand that your data has to

be secure especially in today's world so

there are three steps which are involved

when you're working with data security

in Hadoop the first most important step

is that you have to create a channel for

data flow and if the channel already

exists you need to secure this channel

this channel we're talking about is the

entity which connects your client to the

server and second thing after you have

authenticated the channel after you've

secured the channel the second step

involves the clients making use of

something called as a stamp so this

stamp is basically used and received to

create something called a service

request now this is done to ensure that

it is the actual client who is

requesting the data and not someone else

in their place and this adds legitimacy

to the client and the server can see

this via the stamp and after which

basically is a very simple step where

these clients make use of something

called as a service ticket and this

service ticket is basically a tool which

is used to authenticate the server and

respond back to the client as well so

the usage of stamps and service tickets

are very important in the concept of

data security coming to question number

23 this is a fairly short question but

ours is again very very important that

you understand this as well so what are

the default port numbers for port

tracker task tracker and name node in

Hadoop well there are three different

port numbers associated with it the job

tracker has the default port five double

zero three zero where task tracker has

the default port five double zero six

zero and the name node has the default

port five zero zero seven zero so make

sure you understand and remember this in

however the best way you find are to

remember this but then again using it

twice or thrice will actually help you

concrete this this particular port

number detail into your brain because at

the end of the day this is something

which will come by practice and you will

not forget it to be honest so coming to

question number 24 well question number

24 is primary concerned with respect to

the revenue of the company it states

I will big data analytics help my

company increase its revenue well this

is a question that you can answer in

multiple ways there is no one set answer

that you can have for this but then here

are some of the important things that

you can say in the world where we driven

by data making effective use of it is

basically the entity that drives between

success and failure so effective use of

data is very important and when data is

used effectively it will directly

correlate to having structured growth in

the company and then with respect to big

data it is used to drive customer value

and it will ensure that your customer

retention rate increases at the same

time as well and you can perform a

variety of things among which one

important thing is something called as

manpower forecasting and this is

basically used to understand how the

human resources are being effectively

put to use in the company as well and

this again will create improvised

methodologies for human resource

management and staffing methodologies as

well and the most important point that

you can highlight on is the fact that

big data analytics will bring down the

production costs in a exponential way

because this is why big data has been

put into use in today's world and the

analytics aspect of it is booming since

its launch just because of this it will

make sure that the production cost will

go down rapidly as well so make sure to

mention that and at our halfway point is

question number 25 question number 25 is

concerned with what our data engineer

actually does in in his day-to-day role

well a data engineer is responsible to

handle the inflow of information and

creation of process pipelines so a data

engineer will sit alongside a data

architect to do this which will be

checking out in another question and

then a data engineer

is responsible for maintaining the data

staging areas he's responsible for ETL

data transformations entity

transformations basically and then and

then another very important aspect of a

data engineers world is the ability to

perform data cleaning and removal of

noise removal of redundancies or removal

of any

which way which might not be useful in

converting the raw data into useful

information because data because if the

data is not clean it will lead to very

unofficial outputs especially when

you're performing analytics so make sure

you highlight on the data cleaning

aspect as well and of course as a data

engineer it is expected that you have

the ability to create very good queries

when you're working with up when you're

working with any sort of data operations

because it will majorly involve a lot to

do with data extraction and working with

that as well so coming to question

number 26 so question number 26 goes

like this what are some of the

technologies and skills that a data

engineer should possess the interviewer

at this point of time could be asking

you this question to see if you have

understood the entirety of the role of a

data engineer so some of the very most

important skills and technologies that a

data engineer must have is of course

starting with mathematics the concepts

of probability and linear algebra have a

lot of weight is when you're applying

for a data engineer role and you need to

work with statistics concepts of machine

learning which can again be achieved

when you're working with programming

languages such as Python R or even SAS

as well and since you're working with a

lot of data handling entities again

Hadoop forms to be a very vital aspect

of a data engineer working with SQL and

high fql high fql is very similar to SQL

which is basically the querying language

which is used by a tool called as hive

which again will be checking out in the

next couple of questions so make sure to

name the technologies and the skills

that you think of course you can add on

more to this and eventually create a

list of your own as well and with that

or we can come to question number 27 I

just mentioned data architect a couple

of questions ago so what is the

difference between a data architect and

a data engineer well a data architect is

a person who is mainly responsible for

managing all of the data that comes into

the organization basically so whenever

you are talking about data entry think

of this data can come from Facebook it

can come from Twitter it can come from a

local storage it can come from your

cloud storage it can come from an

entirely different network it can come

from

you know a search result of whatever it

is so when you're working with Big Data

the most important aspect is the variety

of data and the ability of the data

architect to handle the variety of the

data so so data architect is primarily

concerned with the implementation of

this new data into your own architecture

where it might create some conflicts as

well so how can these conflicts be

cleared in a way the pipeline is in a

way that the pipeline is very smooth for

the inflow of data and then of the data

engineer comes into picture so basically

the data engineer is primarily

responsible to work with the data

architect in actually setting up and

establishing this pipeline we call it

the data warehousing pipeline and it can

be well put together with the help of a

data architect and a data engineer and

at the end of it this will also result

in the creation of data hubs data

processing methodologies and some of the

custom protocols which are you know

which are basically required for the

working of that particular architecture

as well so this forms to be the basic

difference between a data architect and

a data engineer this brings us to

question number 28 so how is the

distance or between each of the

different nodes in the distributed

architecture defined whenever a person

uses Hadoop well make sure you explain

on what nodes are and how nodes are

scaled across whenever you think of

approaching this particular answer the

nodes are kept in such a way that there

is a distance between them and with

Hadoop it makes it very easy to assess

and find this distance because it is a

very simple sum of the distance between

your current node and the node that you

want to find the distance to instead of

doing the calculations Maili with Hadoop

as I just mentioned it gives you the

gate distance method and this method can

be put to use effectively to basically

calculate all of the distances so the

simplest answer to so how one can find

the distance between the nodes in Hadoop

is to basically use the gate distance

method so make sure are you emphasize on

that as well but then it is always

advantageous to mention the manual

working of it in case if they ask which

is basically finding the sum of the

distance R between all of the closest

corresponding nodes which

exist so with that we come to question

number 29 so question number 29 states

what is the data that was actually

stored in the name Lord as I mentioned

previously named notice responsible for

having the data with respect to the

actual data that you're working with

what I mean is so basically this is

called as metadata where data is again

describing another piece of data so

metadata information is stored in name

node which corresponds to all of the

actual block data which is present in

the data node so name node is this

descriptor file that you can consider

about the actual data being present in

the data nodes and and it is as simple

as that and with that we come to

question number 13 question number 30

what is meant by rock awareness well

wrack awareness is again a very widely

used concept these days and this

question is again very high and this

question is highly probable to be asked

in the interview as well why do I say

this because wrack awareness is

something which is really nice it is a

concept in which the name node actually

goes on to use the data node or you know

to directly increase all of the incoming

network traffic into that particular

distributed architecture as well so what

it basically does is that whenever there

is any read operation or many any write

operation that is being performed there

is a rank which is associated to each of

these operations and so whenever a read

or write operation is basically created

there is a rack which goes into that

operation be it a read operation or a

write operation so it is executed in a

way where you notice it is the closest

rack to which the data access was

performed through so whenever you talk

about rack awareness so basically it is

basically telling that Hadoop

architecture makes use of this to

increase your traffic by performing

operations in parallel and telling her

dupe that it is doing so so this is a

very simple explanation of rack

awareness whenever we talk about

communication in Hadoop a very common

question that they can ask you is what

is meant by the heartbeat message we

already checked out that heartbeat is

one of the two ways which is basically

used or to communicate between the name

node and

the data node but then you need to

understand that heartbeat is a very

important signal which is sent by the

data node so as literally the name

suggests heartbeat is basically the data

node telling the name node that it's

still operational and then it is still

working fine if there is no heartbeat

message sent from the data in order to

the name node the name Lord thinks that

this particular data aspect is corrupted

or it doesn't or it isn't operational so

a heartbeat is literally used to track

if the data node is functioning or not

and it is as simple as that this brings

us to question number 32 it states what

is the use of a context object in Hadoop

well a context object is used in Hadoop

and it is used together with something

called as the mapper class and this

combination with the mapper class and

the contest and the context object

basically creates a path for

communication so this is very important

because in Hadoop or any distributed

architecture in the field of data

engineering is where data communicates

with a lot of other entities with the

context object it makes it very easy to

understand what the system configuration

is what are the jobs that are supposed

to be executed and the details

corresponding to the job as well so

these to form to be the very vital use

of context object but of course you can

also state that alongside these context

object is actually used to send

informations to certain methods or you

know these methods can be the set of

method the map method and even the

cleanup method that we already checked

out so there is a wide variety of usage

whenever one talks about context objects

in Hadoop this brings us to question

number 33 question number 33 states what

is the use of hive in the Hadoop

ecosystem well as I've already mentioned

before hive is one of the very important

tools set up that is in the Hadoop

architecture which is basically used to

provide the user with an interface so

this interface is used to handle and

work with the data which is actually

stored think of it like a database

management system but here we are

talking about a distributed architecture

as I've mentioned previously our hive

query languages are very similar to the

working of SQL

languages and these are executed to be

be converted into the MapReduce jobs

which actually perform the data

manipulation there so you actually write

a query in hive which is then converted

into a MapReduce job and in the

MapReduce job the data actually gets

processed so this is how you can handle

all of the complexity which comes

whenever you have to work with multiple

MapReduce jobs at a single time and with

respect to hive it gives you a user

interface to simplify all of this to an

exponential level and with this we can

check out question number 34 so question

number 34 States what is the use of meta

store in hive

well so meta store is a very simple

entity it is basically used as a place

where you can store your schemas and

your hive tables that's it so whenever

you asked about meta store make sure you

explain it in a simple way and not

complicated it is a storage location

which is used to store the schemas and

the hive tables so what does the data

that gets actually stored you know the

various mappings in between the data

entities the various definitions which

define the relationships or even the

data and such as metadata can be stored

in the meta store as well and of course

after all of the data is stored into the

meta store this goes into the our DBMS

or wherever it is required and then used

as per the application so with this you

can already understand that meta store

is very vital to be used when you're

working with hive and this brings us to

question number 35 what are the

components that are available in the

hive data model there are three main

components which are present whenever

you talk about hype it's basically

buckets its tables and its partition

whenever the interviewer asks this there

is no strong requirement that you have

to explain on the working of the

components but then make sure that you

understand and know what these

components does because if they ask a

follow-up question based on the

competence of hive you can answer them

easily as well now coming to a question

of a 36 can you create more than one

table for every data file so or it can

also be asked as you know is it possible

to create a single table for an

individual data file when you work

with Hadoop the simple answer to this

question is yes it is more than possible

to create one single table which

contains data for a data file because in

hive as I've already mentioned in the

previous question all of the schemas get

stored in the meta store so there's

already a structured aspect to how data

is mapped and stored by making use of a

single table it even simplifies it

further down rather than the already

simple existing model of the meta store

so this makes it very easy to actually

go on to or extract the data or extract

the analytics aspect of the data

whenever required as well and with this

we come to question number 37 question

number 37 states what is the meaning of

skewed tables in hive this is a very

very common question whenever the

interviewer asks about hive weaker

skewed tables are the entities that are

present in hive where all of the columns

or the rows can contain data which is

very much repeated you know so if you

hive table consists of a lot of numbers

let's say on a simplified example so

here the numbers are repeated a lot so

if there is a lot of repetition in the

data that's present in your tables more

the skewness of the table so a skew

table is basically a table which will

have a repeated set of values present

inside them whenever using when were you

using hive the table is actually

considered as skewed while creating it

itself make sure you highlight on this

point if you already know that your

table will contain repeated information

you can classify the table specifically

as skewed whenever you are creating it

and basically by doing this it ensures

that you know all of the values can be

written in two separate file to avoid

data redundancy and later these files

which are not redundant can go into a

same file so as I just mentioned this is

used - this is used as a structured way

to approach the data and to effectively

store them as well so one important

takeaway from this answer for you guys

is that if the data is more repeated in

the table it is more skewed so this term

forms to be very important here and then

coming to the next question is

questionable 38 what are the collections

that are present in hive so collections

are nothing

the datatypes of hive so whenever you

are asked about collections understand

that the interviewer is trying to ask

you about the data types so there are

four main ways hive can handle data

through structured aspects it is done

using arrays data is handled using

concept of maps it is handled using

struct and Union so again as I've

mentioned in the previous questions if

the interviewer is it's expecting you to

explicate on this particular question

make sure to talk a little bit about all

of these individual data types and where

they can be best used as well that is

going to add a bit of value to your

candidature as well so coming to

question number 39 what is the meaning

of sir day in hive well sir day is

basically a short form for serialization

and deserialization so whenever data is

mulled across - Able's we have two

operations which are performed one is

the serialization operation and the

other one is the D serialization

operation

whenever serialization occurs so

basically the entity which does this it

is called as a serializer

the serializer will take in all of the

Java objects which comes to it

it converts it into a format which is

understood by the HDFS and after this

HDFS will actually take over completely

and it will ensure that it can be used

for the appropriate storage function so

serialization is the basic conversion of

the input data into a format which is

understood by the HDFS now deserialize

ER is basically taking any record which

is present in the HDFS and converting it

back into a Java object so this is

basically done - to help I understand

what the data actually means so again D

serializer will basically take your

record and convert it back into a Java

object to make sure hive understands

about the data is after the

serialization operation hive will not be

able to understand what the data is

hence the requirement for the D

serializer and with this we come to

question number 40 so question number 40

is concerned with what the table

creation functions are that are present

in hive well there are four main

important table creation functions that

are present in hive so there the explore

function when you're working with

explore function of course when you're

working with maps there's a JSON or

disco tuple function and there's a stack

function as well so these are the four

functions which are primarily used for

table creation whenever you're working

with Hayek's so make sure you coat these

four functions and then moving on to

question number 41 question number 41

states what is the role of the dot v RC

function in hi it can also be called as

a dot HIV e rc file in case if the

interviewer wants to separate it out and

tell you but then it's called as the hi

RC file in general so what is the role

of this particular file the first

important part of your answer should be

that this is used for initialization so

whenever you want to write any piece of

code for hive right you first open up

the entity which is of course the

command line interface and whenever the

command line interface is opened this

hive RC file is the first file you have

to load in case if you have to work with

hive as I just mentioned so what this

file contains is basically all of the

parameters that you will have to

initially set to work with your hive

model as well so this forms a very

important aspect to tell the interviewer

that it's used for initialization when

you're working with hive and it is one

of the first commands that you will put

into the command line interface before

working with the files as well and why

is it done it is done to basically set

all of the parameters before beginning

the work in - coming to question number

42 so what are arcs and K works whenever

you walking with data engineering

aspects well again this is a very simple

question with a very simple answer but

then this is very much important as it

is asked in most of the data engineering

interviews out there so the arcs

function is basically as the name

suggests is the argument function it is

used to it is used to define a set

ordered function which is basically used

in the command line so let's say you

have multiple functions you want to

execute on the command line the arcs

function is basically used to define all

of these ordered functions to be used in

the command line interface so coming to

the Kay box function or it's the kW arcs

function is basically trying to denote

that there are certain arguments that

are unorganized that are unordered and

these are used alongside and these are

used alongside as the input to a

function so your arcs function is to

basically denote a creation of an

ordered function but your kW arcs

function is basically used to denote the

set of arguments that are basically

unordered and these go into the function

as well so this is the simple

understanding of what arcs and kW arcs

mean and with this we come to question

number 43

how can you see the structure of a

database by using MySQL well it is very

simple the syntax to understand and see

the structure of a database is to

describe the database so to describe the

database in MySQL you have a very simple

command called as describe itself as the

name suggests

so describes space table name and of

course a semicolon at the end we'll give

you important aspects of that particular

table in that database when you're

working with MySQL as well so make sure

to write down the syntax and of course

you can give an example as well by

creating a table and show how its

described when the described

table name syntax is used as well so

enough question number 44 States can you

search for a specific string in a column

which is present in a MySQL table so can

you search for something which is

specific to the name of that column or

the data in that column you know

whenever there is a MySQL table is

another way or this question can be

asked so the simple answer to this is

yes because whenever you're working with

MySQL you can find any specific string

you require any substring and you can

perform operations on this easily by

making use of the regular expression

operator so the short form of the

regular expression operator is a reg X

and reg X is basically used to do

exactly this and with this we move on to

question number 45 so question number 45

deals with asking you the difference

between a data warehouse and a database

well this is a very important question

so make sure to keep this answer very

concise and in a in an efficient manner

so let's begin so basically whenever

we'll be beginning with data warehousing

it is the end the entire focus of data

warehousing is to make use of certain

functions called as

functions so aggregation functions are

basically min max average sum difference

all of these functions and these

functions are used to perform certain

set of calculations and you'd be

selecting some sort of data to perform

processing so this is the goal of data

warehousing now whenever we talk with

databases databases is concerned with

more because you're you'll be talking

about how the data is input so how the

data is put into the database how you

can manipulate the data how you can

perform certain operations where you're

modifying it you're deleting it and much

more so a database is concerned with

speed and efficiency because data access

data processing and data storage is

happening right here and then the

difference is actually as simple as this

as stated so make sure to answer this so

make sure to answer this in the concise

way coming to question number 46

question number 46 has a lot of weight

edge because this could be you know in

the top three questions that you

guarantee be asked in any of the

interviews it states have you earned any

sort of certification to boost your

opportunities as a data engineer so

whenever your interviewer asks you this

question he or she is trying to find out

if you are really interested in the

individual that you are applying for so

if you say yes to this answer the

interviewer will understand that you

want to enhance and advance your career

in this particular field because you

have put in a lot of time you've put in

some efforts you've learnt the concepts

and you've implemented them actively now

it will also it'll also give an

impression that you are a strong aspirer

and that you are you're you're capable

of learning new things and effectively

putting those to use as well this again

adds to the third point on your screen

that you can see as you being an

effective learner and then one thing you

can talk about it is not just what

you've done in your certification but

then explain about how you have actually

put it into practical use so whenever

you have learned something new the most

important aspect of it is to actually

use it so all the projects that you'll

be working on in your certification

programs is basically B is basically a

real-life project we are solving a

problem so make sure to explain the

problem that you have tried to solve and

explain the approach that you have taken

to solve the problem so this is very

important and if you do not have any

sort of certifications that are

absolutely nothing to worry if you're

strong with all the concepts and if you

do not have any certifications

absolutely nothing to worry if you have

it of course I will give you a lot of

weight edge but if you do not have it

just make sure that you understand all

of the concepts thoroughly in a way that

you're establishing thorough contact

with respect to the interviewer and you

know proving your worth with respect to

this concept of data engineering and

this brings us to question number 47 do

you have any experience working in the

same industry as ours before well the

answer to this is dependent on the

company you're applying for because

again it depends on that particular

company's goals aims and what they're

actually doing so so if you have any

previous experience working in the same

industry make sure you answer this to

the best of your abilities and not just

as a yes or no answer but make sure to

elaborate on the previous experience in

case if you have had any and with

respect to all the tools the techniques

that you have actually used as well this

will actually add a lot of weight it's

again because you're telling the

interviewer that you've had industry

level exposure to the same industry as

in the company that you're applying for

and with this we come to question number

48 question number 48 states why are you

applying for this particular data

engineer role in the company so with

this with this the interviewer is trying

to see if you are proficient with your

subject if you understand everything

you've learned and if you can handle all

of the concepts that is required to

handle the large amount of data that's

present in the company basically as a

data engineer you'll be helping to build

a pipeline as we have discussed and

you'll be working with the same pipeline

as well so to answer this it is always

advantageous to to have a complete

understanding of the job description to

understand what is the compensation that

goes with respect to this particular

role you're applying for what are the

details of the company how the company

works and how you know you can bring the

best of yourself to the company so this

last point that I mentioned how you can

bring the best two best of you to the

company where you explain on that is

very very important so answer this to

the best of your ability and again

I mentioned there is no one-step answer

to this but then this is totally

dependent on you and here is the

framework that you can actually use to

answer it this brings us to the last but

not the least question question number

49

what is your plan after joining this

data engineering role well and again

here is my advice do not do not start on

stories so again with this answer I

would like to give you an advice to make

sure you keep it concise please do not

give any long stories as the companies

as the interviewer might not have the

time or the expectation or to hear from

you as well to hear from you in detail

about what you're planning to do because

you have not joined the company yet so

but then it is very important that you

talk about how you are how you will put

an effort to understand how the data

infrastructure is set up in the company

and how you will take part in this

infrastructure either to make it better

to improvise it and then you know work

easily in collaboration with all of the

other members of the team as well so use

this as a fine print to build your

answer and then give it out in a concise

way and then make sure you do not give

unorganized long answers for this

particular question and with this we

come to the last question which is again

very important if you have had

experience in the field of data

engineering so the question is do you

have any prior experience working with

data modeling again if you are

interviewing for an intermediate role

this question will always be asked for

short it will be asked in the beginning

of the interview in fact so make sure

you answer with a yes or a No and if you

are asked this question there is a good

chance that you are applying for an

intermediate level role so do know this

there are two ways to answer this one if

you answer no to this you can have

proficiency in all of the other data

engineering concepts but not data

modelling so if you answer no it is

completely alright you know talk about

data modelling talk about what you

understand by it and how do you plan to

learn it if you answer no if you answer

yes then make sure to talk about the

tools that you've used to perform data

modeling you know there are tools like

Pentaho and informatica which I use just

for data modeling so if your answer is

yes make sure to elaborate on this

particular aspect

to fit and do not vary as I just

mentioned if your answer is no do it

because you might have proficiency in

some other aspects of data engineering

that they're actively looking for so

this brings us to the end of this

session if you have any queries you can

leave a comment down below thank you so

much for watching guys
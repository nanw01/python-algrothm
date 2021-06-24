create table patients(id integer, name varchar(10));
insert into patients(id, name)
values(1, 'Alan');
insert into patients(id, name)
values(2, 'Ben');
insert into patients(id, name)
values(3, 'Chris');
insert into patients(id, name)
values(4, 'Diane');
insert into patients(id, name)
values(5, 'Elliot');
create table transactions(
    id integer,
    procedure_type varchar(10),
    cost decimal(10, 2),
    patient_id integer
);
insert into transactions(id, procedure_type, cost, patient_id)
values(1, 'cleaning', 100.5, 1);
insert into transactions(id, procedure_type, cost, patient_id)
values(2, 'cleaning', 80.2, 2);
insert into transactions(id, procedure_type, cost, patient_id)
values(3, 'extraction', 200.0, 2);
insert into transactions(id, procedure_type, cost, patient_id)
values(4, 'extraction', 150.2, 3);
insert into transactions(id, procedure_type, cost, patient_id)
values(5, 'root canal', 500.0, 4);
insert into transactions(id, procedure_type, cost, patient_id)
values(6, 'root canal', 500.0, 5);
insert into transactions(id, procedure_type, cost, patient_id)
values(7, 'others', 50.2, 5);
create table transaction_comments(transaction_id integer, comment varchar(100));
insert into transaction_comments(transaction_id, comment)
values(1, 'Good');
insert into transaction_comments(transaction_id, comment)
values(4, 'Bad');
insert into transaction_comments(transaction_id, comment)
values(6, '');
insert into transaction_comments(transaction_id, comment)
values(7, null);
select *
from patients;
select *
from transactions;
Select *
from transaction_comments;
-- Questions:
-- 1. Find transaction that has valid comment 
SELECT *
from transaction as t
    left join transaction_comments as tc on t.id = tc.transaction_id
WHERE tc.comment is NOT NULL
    or tc.comment != '';
-- 2. Find transaction that has no row in transaction_comment
SELECT *
from transaction as t
    left join transaction_comments as tc on t.id = tc.transaction_id
WHERE tc.transaction_id is NULL;
-- 3. Write a SQL query to sort the patient name in the order of highest average transaction cost
SELECT p.name
from patients as p
    left join transactions as t on p.id = t.patient_id
group by p.id
order by avg(t.cost) desc;
-- 4. Write a SQL query to find the transaction with the 3rd highest cost
select rank() over(
        or
    )
from transactions as t;
-- 5. Add primary key and foreign key to the transaction table
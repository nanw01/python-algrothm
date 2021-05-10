select id,
    name
from Students
where department_id not in (
        select id
        from Departments
    );
# Write your MySQL query statement below
select s.id,
    s.name
from Students as s
where s.department_id not in (
        select d.id as department_id
        from Departments as d
    )
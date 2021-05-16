# Write your MySQL query statement below
select uni.unique_id,
    emp.name
from Employees emp
    left join EmployeeUNI uni on emp.id = uni.id;
# Write your MySQL query statement below
select eu.unique_id,
    e.name
from Employees as e
    left join EmployeeUNI as eu on eu.id = e.id;
select d.name as Department,
    e.name as Employee,
    e.salary as Salary
from Employee as e
    join department as d on e.departmentid = d.id
where 3 >(
        select count(distinct e2.salary)
        from employee e2
        where e2.salary > e.salary
            and e.DepartmentId = e2.DepartmentId
    );
# Write your MySQL query statement below
select Department,
    Employee,
    Salary
from (
        select d.Name as Department,
            e.Name as Employee,
            e.Salary,
            dense_rank() over(
                partition by e.DepartmentId
                order by e.Salary desc
            ) as num
        from Employee as e
            left join Department as d on e.DepartmentId = d.Id
    ) as ll
where ll.num < 4
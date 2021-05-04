-- # Write your MySQL query statement below
-- select e3.employee_id
-- from Employees e1,
--     Employees e2,
--     Employees e3
-- where e1.manager_id = 1
--     and e2.manager_id = e1.employee_id
--     and e3.manager_id = e2.employee_id
--     and e3.employee_id != 1
--
-- select employee_id EMPLOYEE_ID
-- from employees
-- where manager_id=1 and employee_id!=1
-- union
-- select a1.employee_id
-- from employees a1,
--     (select employee_id
--     from employees
--     where manager_id=1 and employee_id!=1) a
-- where manager_id=a.employee_id
-- union
-- select a2.employee_id
-- from employees a2,
--     (select a1.employee_id employee_id
--     from employees a1,
--         (select employee_id
--         from employees
--         where manager_id=1 and employee_id!=1) a
--     where manager_id=a.employee_id) a3
-- where manager_id=a3.employee_id
-- order by employee_id;
--
select e1.employee_id
from Employees e1
    left join Employees e2 on e1.manager_id = e2.employee_id
    left join Employees e3 on e2.manager_id = e3.employee_id
where e3.manager_id = 1
    and e1.employee_id <> 1;
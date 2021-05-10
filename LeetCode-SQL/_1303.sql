-- select employee_id,
--     team_size
-- from Employee e
--     left join (
--         select team_id,
--             count(distinct(employee_id)) as team_size
--         from Employee
--         group by team_id
--     ) as t on e.team_id = t.team_id;
# Write your MySQL query statement below
select e1.employee_id,
    e2.team_size
from employee as e1,
    (
        select e.team_id as team_id,
            count(e.employee_id) as team_size
        from Employee as e
        group by e.team_id
    ) as e2
where e1.team_id = e2.team_id
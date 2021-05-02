select project_id
from Project
group by project_id
having count(employee_id) = (
        select count(employee_id)
        from Project
        group by project_id
        order by count(employee_id) desc
        limit 1
    );
# Write your MySQL query statement below
select p.project_id
from project as p
group by p.project_id
having count(p.employee_id) = (
        select count(employee_id)
        from Project
        group by project_id
        order by count(employee_id) desc
        limit 1
    )
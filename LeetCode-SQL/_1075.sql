select project_id,
    round(avg(experience_years), 2) as average_years
from Project
    join Employee using (employee_id)
group by project_id
order by project_id;
# Write your MySQL query statement below
select p.project_id,
    round(avg(e.experience_years), 2) as average_years
from project as p
    left join Employee as e using(employee_id)
group by p.project_id
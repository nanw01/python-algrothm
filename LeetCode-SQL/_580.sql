# Write your MySQL query statement below
select d.dept_name,
    count(s.student_id) as student_number
from department as d
    left join student as s using(dept_id)
group by d.dept_id
order by student_number desc,
    d.dept_name;
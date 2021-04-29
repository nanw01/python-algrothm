# Write your MySQL query statement below
select c.name
from customer as c
where c.referee_id != 2
    or c.referee_id is null;
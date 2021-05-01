select id,
    movie,
    description,
    rating
from cinema
where mod (id, 2) = 1
    and description not like 'boring'
order by rating desc;
# Write your MySQL query statement below
select id,
    movie,
    description,
    rating
from cinema
where mod(id, 2) = 1
    and description not like 'boring'
order by rating desc
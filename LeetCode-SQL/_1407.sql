--# Write your MySQL query statement below
--credit: https://leetcode.com/problems/top-travellers/discuss/572803/MySQL-Simple-Solution
select name,
    sum(ifnull(distance, 0)) as travelled_distance
from rides r
    right join users u on r.user_id = u.id
group by name
order by 2 desc,
    1 asc;
--
--
# Write your MySQL query statement below
select u.name,
    sum(ifnull(r.distance, 0)) as travelled_distance
from Users as u
    left join Rides as r on u.id = r.user_id
group by u.name
order by travelled_distance desc,
    name asc;
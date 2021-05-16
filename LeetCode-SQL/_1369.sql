# Write your MySQL query statement below
select *
from UserActivity
group by username
having count(*) = 1
union all
select u1.*
from UserActivity as u1
    left join UserActivity as u2 on u1.username = u2.username
    and u1.endDate < u2.endDate
group by u1.username,
    u1.endDate
having count(u2.endDate) = 1
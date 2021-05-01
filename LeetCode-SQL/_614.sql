# Write your MySQL query statement below
select f1.follower as follower,
    count(distinct f2.follower) as num
from follow as f1
    inner join follow as f2 on f1.follower = f2.followee
group by f1.follower;
# Write your MySQL query statement below
select f.followee as follower,
    count(distinct f.follower) as num
from follow as f
group by f.followee
having f.followee in (
        select f2.follower
        from follow as f2
    )
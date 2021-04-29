# Write your MySQL query statement below
select u.id as id,
    count(*) as num
from (
        (
            select requester_id id
            from request_accepted
        )
        union all
        (
            select accepter_id id
            from request_accepted
        )
    ) as u
group by u.id
order by num desc
limit 1
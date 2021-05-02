select seller_id
from Sales as seller_id
group by seller_id
having sum(price) = (
        select sum(price) as total_price
        from Sales
        group by seller_id
        order by total_price desc
        limit 1
    );
# Write your MySQL query statement below
select s0.seller_id
from Sales as s0
group by seller_id
having sum(s0.price) = (
        select sum(s.price) as total
        from Sales as s
        group by s.seller_id
        order by total desc
        limit 1
    )
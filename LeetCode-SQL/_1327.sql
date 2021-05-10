--# Write your MySQL query statement below
--credit: https://leetcode.com/problems/list-the-products-ordered-in-a-period/discuss/491314/MYSQL
-- select a.product_name,
--     sum(unit) as unit
-- from Products a
--     left join Orders b on a.product_id = b.product_id
-- where b.order_date between '2020-02-01' and '2020-02-29'
-- group by a.product_id
- -
having sum(unit) >= 100;
# Write your MySQL query statement below
select p.product_name as product_name,
    sum(o.unit) as unit
from Products as p
    join Orders as o on p.product_id = o.product_id
    and o.order_date >= '2020-02-01'
    and o.order_date <= '2020-02-29'
group by p.product_id
having sum(o.unit) >= 100
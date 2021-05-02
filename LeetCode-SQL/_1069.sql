SELECT product_id,
    SUM(quantity) AS total_quantity
FROM Sales
GROUP BY product_id;
# Write your MySQL query statement below
select s.product_id,
    sum(s.quantity) as total_quantity
from sales as s
group by product_id
# Write your MySQL query statement below
select lower(trim(product_name)) as product_name,
    left(sale_date, 7) as sale_date,
    count(lower(trim(product_name))) as total
from Sales
group by lower(trim(product_name)),
    left(sale_date, 7)
order by lower(trim(product_name)) asc,
    left(sale_date, 7) asc;
-- 
-- 
# Write your MySQL query statement below
select lower(trim(product_name)) as product_name,
    date_format(sale_date, "%Y-%m") as sale_date,
    count(lower(trim(product_name))) as total
from Sales
group by lower(trim(product_name)),
    date_format(sale_date, "%Y-%m")
order by lower(trim(product_name)) asc,
    date_format(sale_date, "%Y-%m") asc;
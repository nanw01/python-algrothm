-- 累加
select date,
    count(user_id)
from users_joined
group by date
order by date;
select date,
    count(user_id) as count,
    sum(count(user_id)) over (
        order by date
    ) as running_total
from users_joined
group by date
order by date;
-- moving average 滑动平均
select quarter,
    revenue,
    avg(revenue) over (
        order by quarter rows between 3 preceding and current row
    )
from amazon_revenue;
-- Weighted Moving Average in SQL
select quarter,
    revenue,
    row_number() over ()
from amazon_revenue;
-- 
with t as (
    select quarter,
        revenue,
        row_number() over ()
    from amazon_revenue
)
select t.quarter,
    t.row_number as row_number,
    t2.quarter as quarter_2,
    t2.row_number as row_number_2
from t
    join t t2 on t2.row_number between t.row_number - 3 and t.row_number;
-- SQL case to use the fractional weights
with t as (
    select quarter,
        revenue,
        row_number() over ()
    from amazon_revenue
)
select t.quarter,
    avg(t.revenue) as revenue,
    sum(
        case
            when t.row_number - t2.row_number = 0 then 0.4 * t2.revenue
            when t.row_number - t2.row_number = 1 then 0.3 * t2.revenue
            when t.row_number - t2.row_number = 2 then 0.2 * t2.revenue
            when t.row_number - t2.row_number = 3 then 0.1 * t2.revenue
        end
    )
from t
    join t t2 on t2.row_number between t.row_number - 3 and t.row_number
group by 1
order by 1;
-- Calculating Difference from Beginning / First Row in SQL
select dt,
    price,
    first_value(price) over ()
from trades;
-- First value in the partition
select dt,
    price,
    first_value(price) over (partition by date_trunc('month', dt))
from trades;
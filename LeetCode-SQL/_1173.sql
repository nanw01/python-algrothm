# Write your MySQL query statement below
select round(100 * sum(tag) / count(id), 2) as immediate_percentage
from (
        select d1.customer_id as id,
            case
                when min(order_date) = min(customer_pref_delivery_date) then 1
                else 0
            end as tag
        from Delivery d1
        group by customer_id
    ) as d2
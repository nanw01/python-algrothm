# Write your MySQL query statement below
select s.name
from salesperson as s
where s.sales_id not in (
                select o.sales_id
                from orders as o
                where o.com_id in (
                                select c.com_id
                                from company as c
                                where c.name = 'RED'
                        )
        )
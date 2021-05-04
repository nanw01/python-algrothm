select a.product_id,
    round(
        sum(a.price * b.units) / sum(b.units),
        2
    ) as average_price
from Prices as a
    join UnitsSold as b on a.product_id = b.product_id
    and (
        b.purchase_date between a.start_date and a.end_date
    )
group by a.product_id;
# Write your MySQL query statement below
select p.product_id,
    round(
        sum(p.price * u.units) / sum(u.units),
        2
    ) as average_price
from Prices as p
    join UnitsSold as u on p.product_id = u.product_id
    and (
        u.purchase_date between p.start_date and p.end_date
    )
group by p.product_id
select round(
        (
            100 * (
                select count(*)
                from Delivery d
                where d.order_date = d.customer_pref_delivery_date
            ) / (
                select count(*)
                from Delivery
            )
        ),
        2
    ) as immediate_percentage;
# Write your MySQL query statement below
select round(
        100 * (
            select count(*)
            from Delivery as d
            where d.order_date = d.customer_pref_delivery_date
        ) / (
            select count(*)
            from Delivery as d1
        ),
        2
    ) as immediate_percentage
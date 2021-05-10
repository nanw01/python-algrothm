-- # Write your MySQL query statement below
-- select ad_id,
--     ifnull(
--         round(
--             avg(
--                 case
--                     when action = "Clicked" then 1
--                     when action = "Viewed" then 0
--                     else null
--                 end
--             ) * 100,
--             2
--         ),
--         0
--     ) as ctr
-- from Ads
-- group by ad_id
-- order by ctr desc,
--     ad_id asc;
# Write your MySQL query statement below
select a.ad_id as ad_id,
    ifnull(
        round(
            avg(
                case
                    when a.action = 'Clicked' then 1
                    when a.action = 'Viewed' then 0
                    else null
                end
            ) * 100,
            2
        ),
        0
    ) as ctr
from ads as a
group by a.ad_id
order by ctr desc,
    ad_id
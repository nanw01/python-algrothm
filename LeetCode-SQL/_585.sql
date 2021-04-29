select sum(TIV_2016) as TIV_2016
from insurance a
where (
        1 = (
            select count(*)
            from insurance b
            where a.LAT = b.LAT
                and a.LON = b.LON
        )
        and 1 < (
            select count(*)
            from insurance c
            where a.TIV_2015 = c.TIV_2015
        )
    );
# Write your MySQL query statement below
select sum(TIV_2016) as TIV_2016
from insurance
where TIV_2015 in(
        select TIV_2015
        from insurance
        group by TIV_2015
        having count(*) >= 2
    )
    and (lat, lon) in(
        select lat,
            lon
        from insurance
        group by lat,
            lon
        having count(*) = 1
    )
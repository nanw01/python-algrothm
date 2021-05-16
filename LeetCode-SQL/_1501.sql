select co.name as country
from Calls c
    left join Person p on c.caller_id = p.id
    or c.callee_id = p.id
    left join Country co on left(p.phone_number, 3) = co.country_code
group by co.country_code
having avg(c.duration) > (
        select avg(duration)
        from Calls
    )
SELECT (
        CASE
            WHEN MOD(id, 2) != 0
            AND counts != id THEN id + 1
            WHEN MOD(id, 2) != 0
            AND counts = id THEN id
            ELSE id - 1
        END
    ) AS id,
    student
FROM seat,
    (
        SELECT COUNT(*) AS counts
        FROM seat
    ) AS seat_counts
ORDER BY id ASC;
# Write your MySQL query statement below
select(
        case
            when mod(id, 2) = 1
            and s.counts = id then id
            when mod(id, 2) = 1
            and s.counts != id then id + 1
            else id - 1
        End
    ) as id,
    student
from seat,
    (
        select count(*) as counts
        from seat
    ) as s
order by id
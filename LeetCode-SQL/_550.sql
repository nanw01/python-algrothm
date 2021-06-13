# Write your MySQL query statement below
SELECT ROUND(
        (
            SELECT COUNT(DISTINCT A.player_id)
            FROM Activity AS A
                JOIN Activity AS B ON A.player_id = B.player_id
                AND DATEDIFF(B.event_date, A.event_date) = 1
            WHERE A.event_date = (
                    SELECT MIN(event_date)
                    FROM Activity
                    WHERE player_id = A.player_id
                )
        ) / (
            SELECT COUNT(DISTINCT player_id)
            FROM Activity
        ),
        2
    ) AS `fraction`;
# Write your MySQL query statement below
select round(
        (
            select count(*)
            from activity as a2
            where (a2.player_id, a2.event_date) in (
                    select distinct(a.player_id),
                        min(a.event_date) + interval 1 day as event_date
                    from activity as a
                    group by a.player_id
                )
        ) /(
            select count(distinct a3.player_id)
            from activity as a3
        ),
        2
    ) as fraction
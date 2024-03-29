select s1.gender,
    s1.day,
    sum(s2.score_points) as total
from Scores s1,
    Scores s2
where s1.gender = s2.gender
    and s1.day >= s2.day
group by s1.gender,
    s1.day
order by s1.gender,
    s1.day;
# Write your MySQL query statement below
select s.gender,
    s.day,
    sum(s2.score_points) as total
from Scores as s,
    Scores s2
where s.gender = s2.gender
    and s.day >= s2.day
group by s.day,
    s.gender
order by s.gender,
    s.day
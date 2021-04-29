# Write your MySQL query statement below
select c.name
from Candidate as c
where c.id = (
        select v.CandidateId
        from Vote as v
        group by v.CandidateId
        order by count(v.CandidateId) desc
        limit 1
    )
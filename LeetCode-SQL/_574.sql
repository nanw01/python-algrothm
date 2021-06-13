# Write your MySQL query statement below
select c.name
from Candidate c
where id = (
        select v.CandidateId
        from vote as v
        group by v.candidateid
        order by count(v.CandidateId) desc
        limit 1
    )
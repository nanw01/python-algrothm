SELECT question_id AS 'survey_log'
FROM survey_log
GROUP BY question_id
ORDER BY COUNT(answer_id) / COUNT(
        case
            when survey_log.action = 'show' then survey_log.action
            else null
        end
    ) DESC
LIMIT 0, 1;
# Write your MySQL query statement below
select s.question_id as survey_log
from survey_log as s
group by s.question_id
order by count(s.answer_id) / count(
        case
            when s.action = 'show' then s.action
            else null
        END
    ) DESC
limit 0, 1
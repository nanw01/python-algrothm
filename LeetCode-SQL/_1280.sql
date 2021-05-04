-- select a.student_id,
--     a.student_name,
--     b.subject_name,
--     count(c.subject_name) as attended_exams
-- from Students as a
--     join Subjects as b
--     left join Examinations as c on a.student_id = c.student_id
--     and b.subject_name = c.subject_name
-- group by a.student_id,
--     b.subject_name;
# Write your MySQL query statement below
select st.student_id,
    st.student_name,
    su.subject_name,
    count(e.subject_name) as attended_exams
from students as st
    join Subjects as su
    left join Examinations as e on st.student_id = e.student_id
    and su.subject_name = e.subject_name
group by st.student_id,
    su.subject_name
order by st.student_id,
    su.subject_name
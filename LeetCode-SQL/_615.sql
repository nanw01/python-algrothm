-- SELECT d1.pay_month,
--      d1.department_id,
--      CASE
- -
WHEN d1.department_avg > c1.company_avg THEN 'higher' --           WHEN d1.department_avg < c1.company_avg THEN 'lower'
--           ELSE 'same'
--      END AS 'comparison'
-- FROM (
--           (
--                SELECT LEFT(s1.pay_date, 7) pay_month,
--                     e1.department_id,
--                     AVG(s1.amount) department_avg
--                FROM salary s1
--                     JOIN employee e1 ON s1.employee_id = e1.employee_id
--                GROUP BY pay_month,
--                     e1.department_id
--           ) d1
--           LEFT JOIN (
--                SELECT LEFT(pay_date, 7) pay_month,
--                     AVG(amount) company_avg
--                FROM salary
--                GROUP BY pay_month
--           ) c1 ON d1.pay_month = c1.pay_month
--      )
-- ORDER BY pay_month DESC,
--      department_id;
# Write your MySQL query statement below
select pay_month,
     department_id,
     case
          when d_avg > c_avg then 'higher'
          when d_avg = c_avg then 'same'
          when d_avg < c_avg then 'lower'
     end as comparison
from (
          (
               select left(s.pay_date, 7) as pay_month,
                    e.department_id as department_id,
                    avg(s.amount) as d_avg
               from salary as s
                    left join employee as e using(employee_id)
               group by pay_month,
                    e.department_id
          ) as ss1
          left join (
               select left(s.pay_date, 7) as pay_month,
                    avg(s.amount) as c_avg
               from salary as s
               group by pay_month
          ) as ss2 using(pay_month)
     )
ORDER BY pay_month DESC,
     department_id;
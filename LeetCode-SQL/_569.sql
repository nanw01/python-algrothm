# Write your MySQL query statement below
select e1.id,
    e1.Company,
    e1.Salary
from Employee e1,
    Employee e2
where e1.Company = e2.Company
group by e1.Company,
    e1.Salary
having sum(
        case
            when e1.Salary = e2.Salary then 1
            else 0
        end
    ) >= abs(sum(sign(e1.Salary - e2.Salary)))
order by e1.Id;
--
--
--
# Write your MySQL query statement below
select t1.Id,
    t1.Company,
    t1.Salary
from(
        select Id,
            Company,
            Salary,
            row_number() over(
                partition by Company
                order by Salary
            ) rownum,
            count(1) over(partition by Company) num
        from Employee
    ) t1
where t1.rownum >= num / 2
    and t1.rownum <= num / 2 + 1;
--
--
# Write your MySQL query statement below
select ee.id,
    ee.company,
    ee.salary
from (
        select e.id,
            e.company,
            e.salary,
            row_number() over(
                partition by e.company
                order by e.salary
            ) as row_num,
            count(e.id) over(partition by e.company) as num
        from Employee as e
    ) as ee
where ee.row_num >= ee.num / 2
    and ee.row_num <= ee.num / 2 + 1
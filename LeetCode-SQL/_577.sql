select name,
    bonus
from employee
    left join bonus using(empId)
having bonus < 1000
    or bonus is null
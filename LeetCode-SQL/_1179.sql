select id,
    (
        case
            when month = 'Jan' then revenue
            else null
        end
    ) as 'Jan_Revenue',
    max(
        case
            when month = 'Feb' then revenue
            else null
        end
    ) as 'Feb_Revenue',
    max(
        case
            when month = 'Mar' then revenue
            else null
        end
    ) as 'Mar_Revenue',
    max(
        case
            when month = 'Apr' then revenue
            else null
        end
    ) as 'Apr_Revenue',
    max(
        case
            when month = 'May' then revenue
            else null
        end
    ) as 'May_Revenue',
    max(
        case
            when month = 'Jun' then revenue
            else null
        end
    ) as 'Jun_Revenue',
    max(
        case
            when month = 'Jul' then revenue
            else null
        end
    ) as 'Jul_Revenue',
    max(
        case
            when month = 'Aug' then revenue
            else null
        end
    ) as 'Aug_Revenue',
    max(
        case
            when month = 'Sep' then revenue
            else null
        end
    ) as 'Sep_Revenue',
    max(
        case
            when month = 'Oct' then revenue
            else null
        end
    ) as 'Oct_Revenue',
    max(
        case
            when month = 'Nov' then revenue
            else null
        end
    ) as 'Nov_Revenue',
    max(
        case
            when month = 'Dec' then revenue
            else null
        end
    ) as 'Dec_Revenue'
from Department
group by id;
# Write your MySQL query statement below
select id,
    Max(
        case
            when d.month = 'Jan' then d.revenue
            else null
        end
    ) as Jan_Revenue,
    Max(
        case
            when d.month = 'Feb' then d.revenue
            else null
        end
    ) as Feb_Revenue,
    Max(
        case
            when d.month = 'Mar' then d.revenue
            else null
        end
    ) as Mar_Revenue,
    Max(
        case
            when d.month = 'Apr' then d.revenue
            else null
        end
    ) as Apr_Revenue,
    Max(
        case
            when d.month = 'May' then d.revenue
            else null
        end
    ) as May_Revenue,
    Max(
        case
            when d.month = 'Jun' then d.revenue
            else null
        end
    ) as Jun_Revenue,
    Max(
        case
            when d.month = 'Jul' then d.revenue
            else null
        end
    ) as Jul_Revenue,
    Max(
        case
            when d.month = 'Aug' then d.revenue
            else null
        end
    ) as Aug_Revenue,
    Max(
        case
            when d.month = 'Sep' then d.revenue
            else null
        end
    ) as Sep_Revenue,
    Max(
        case
            when d.month = 'Oct' then d.revenue
            else null
        end
    ) as Oct_Revenue,
    Max(
        case
            when d.month = 'Nov' then d.revenue
            else null
        end
    ) as Nov_Revenue,
    Max(
        case
            when d.month = 'Dec' then d.revenue
            else null
        end
    ) as Dec_Revenue
from Department as d
group by id
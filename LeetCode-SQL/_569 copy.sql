# Write your MySQL query statement below
SELECT E.Company,
    FLOOR((COUNT(*) -1) / 2) AS `beg`,
    if(COUNT(*) % 2 = 1, 0, 1) AS `cnt`
FROM employee AS E
GROUP BY E.Company
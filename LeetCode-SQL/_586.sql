SELECT o.customer_number
FROM orders as o
GROUP BY o.customer_number
ORDER BY COUNT(*) DESC
LIMIT 1;
select Name as Customers
from Customers
    left join Orders on Customers.Id = Orders.CustomerId
where Orders.CustomerId is Null;
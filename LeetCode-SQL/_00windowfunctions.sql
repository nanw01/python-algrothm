select name,
    sum(weight) over (
        order by weight DESC ROWS between unbounded preceding and current row
    ) as running_total_weight
from cats
order by running_total_weight;
-- 
select name,
    weight,
    breed,
    coalesce(
        cast(
            lead(weight, 1) over (
                partition by breed
                order by weight
            ) as varchar
        ),
        'fattest cat'
    ) as next_heaviest
from cats
order by weight;
-- Question: The cats have decided the correct weight is the same as the 4th lightest cat.All cats shall have this weight.
-- Except in a fit of jealous rage they decide to
-- set the weight of the lightest three to 99.9 Print a list of cats,
--     their weights
--     and their imagined weight Return: name,
--     weight,
--     imagined_weight
-- Order by: weight
select name,
    weight,
    coalesce(
        nth_value(weight, 4) over (
            order by weight
        ),
        99.9
    ) as imagined_weight
from cats
order by weight;
-- Question: The cats want to show their weight by breed.The cats agree that they should show the second lightest cat 's weight (so as not to make other cats feel bad)
-- Print a list of breeds, and the second lightest weight of that breed
-- Return: breed, imagined_weight
-- Order by: breed
select distinct breed,
    nth_value(weight, 2) over(
        partition by breed
        order by weight RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as imagined_weight
from cats
order by breed;
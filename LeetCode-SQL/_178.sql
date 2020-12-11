select 
      score
      , dense_rank() over(order by score desc) as "Rank"
from scores

-- 
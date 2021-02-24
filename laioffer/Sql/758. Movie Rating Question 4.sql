SELECT distinct re.name
from reviewer re,
    rating ra
where re.rid = ra.rid
    and ra.ratingdate is Null
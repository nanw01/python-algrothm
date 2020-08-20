
# Class Objects

# Python isn't strongly typed, so we can save anything we like:
# just as we stored a tuple of (priority,thing) in previous section.
# We can also store class objects if we override __cmp__() method:


# Override __lt__ in Python 3, __cmp__ only in Python 2

import heapq


class Skill(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print('New Level:', description)
        return

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return str(self.priority) + ": " + self.description


s1 = Skill(5, 'Proficient')
s2 = Skill(10, 'Expert')
s3 = Skill(1, 'Novice')

l = [s1, s2, s3]

heapq.heapify(l)
print("The 3 largest numbers in list are : ", end="")
print(heapq.nlargest(3, l))

while l:
    item = heapq.heappop(l)
    print(item)

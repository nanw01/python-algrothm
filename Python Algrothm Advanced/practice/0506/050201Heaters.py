# ### Ex.1: Heaters
# Winter is coming! Your first job during the contest is to design a standard heater with fixed warm radius to warm all the houses.
# Now, you are given positions of houses and heaters on a horizontal line, find out minimum radius of heaters so that all houses could be covered by those heaters.
# So, your input will be the positions of houses and heaters seperately, and your expected output will be the minimum radius standard of heaters.
# ** Note: **
# Numbers of houses and heaters you are given are non-negative and will not exceed 25000.
# Positions of houses and heaters you are given are non-negative and will not exceed 10^9.
# As long as a house is in the heaters' warm radius range, it can be warmed.
# All the heaters follow your radius standard and the warm radius will the same.
# ** Example 1: **
# Input: [1,2,3],[2]
# Output: 1
# Explanation: The only heater was placed in the position 2, and if we use the radius 1 standard, then all the houses can be warmed.
# ** Example 2: **
# Input: [1,2,3,4],[1,4]
# Output: 1
# Explanation: The two heater was placed in the position 1 and 4. We need to use radius 1 standard, then all the houses can be warmed.


# 将房子位置插入到暖气当中，反向思维


from bisect import bisect


def findRadius(house,heaters):
    heaters.sort()
    ans = 0

    for h in house:
        hi = bisect(heaters,h)
        left = heaters[hi-1] if hi - 1>=0 else float('-inf')
        right = heaters[hi] if hi<len(heaters) else float('inf')
        ans = max(ans,min(h-left,right-h))
    
    return ans

house = [1,12,23,34,99]
heaters = [12,24]

print(findRadius(house,heaters))





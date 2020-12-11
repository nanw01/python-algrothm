# ### Ex.6 Jewels and Stones 
# You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.
# The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".
# Example 1:
# Input: J = "aA", S = "aAAbbbb"
# Output: 3
# Example 2:
# Input: J = "z", S = "ZZ"
# Output: 0
# Note:
# S and J will consist of letters and have length at most 50.
# The characters in J are distinct.

## brute force
def numJewelsInStones_bf(J, S):
    count=0
    for c in S:
        if c in J:
            count += 1
    return count

## set
def numJewelsInStones(J, S):
    setJ = set(J)
    return sum(s in setJ for s in S)


J = "aA"
S = "aAAbbbb"

print(numJewelsInStones_bf(J, S))

print(numJewelsInStones(J, S))
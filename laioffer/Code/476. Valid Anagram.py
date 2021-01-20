class Solution(object):
    def isAnagram(self, source, target):
        """
        input: string source, string target
        return: boolean
        """
        # write your solution here

        return sorted(source) == sorted(target)


# For example,
# s = "anagram", t = "nagaram", return true.
# s = "rat", t = "car", return false.

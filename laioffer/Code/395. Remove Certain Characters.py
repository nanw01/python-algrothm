class Solution(object):
    def remove(self, input, t):
        """
        input: string input, string t
        return: string
        """
        # write your solution here
        return ''.join([i for i in input if i not in t])


if __name__ == "__main__":
    s = Solution()
    print(s.remove("abcd", "ab"))

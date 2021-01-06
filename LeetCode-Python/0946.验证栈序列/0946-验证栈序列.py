class Solution(object):
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        s = []
        popped = popped[::-1]
        for num in pushed:
            s.append(num)
            while s and popped and s[-1] == popped[-1]:
                s.pop()
                popped.pop()
        return not s and not popped


if __name__ == "__main__":
    pushed = [585, 760, 754, 150, 225, 358, 219, 933, 851, 804, 860, 184, 341, 317, 670, 696, 38, 437, 726, 353, 365, 683, 107, 997,
              425, 424, 609, 582, 36, 206, 537, 572, 330, 120, 915, 736, 47, 892, 676, 939, 462, 814, 523, 30, 171, 218, 334, 368, 932, 986]
    popped = [150, 225, 358, 219, 754, 760, 860, 341, 184, 804, 851, 933, 38, 696, 726, 683, 609, 424, 582, 206, 537, 36, 572, 425,
              997, 330, 915, 120, 107, 365, 736, 47, 353, 437, 670, 317, 585, 892, 676, 939, 462, 30, 218, 171, 523, 814, 334, 986, 932, 368]
    s = Solution()
    print(s.validateStackSequences(pushed, popped))

class Solution(object):
    def strstr(self, large, small):
        """
        input: string large, string small
        return: int
        """
        # write your solution here

        if len(small) == 0:
            return 0

        for i in range(len(large)-len(small)+1):
            pos = i
            for j in range(len(small)):
                # print(pos, large[pos], j, small[j])
                if large[pos] == small[j] and j == len(small)-1:
                    return i
                elif large[pos] != small[j]:
                    break
                pos += 1
            # print('-------'+str(i)+'---------')
        return -1


print(Solution().strstr('abbaabbab', 'bbab'))

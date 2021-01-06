class Solution(object):
    def reverse(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here
        a = [i for i in range(len(input)) if input[i]
             in ['a', 'e', 'i', 'o', 'u']]
        b = a[::-1]

        input = list(input)
        for i in range(len(a)//2):
            input[a[i]], input[b[i]] = input[b[i]], input[a[i]]

        return ''.join(input)


if __name__ == "__main__":
    s = Solution()
    print(s.reverse('abbegi'))

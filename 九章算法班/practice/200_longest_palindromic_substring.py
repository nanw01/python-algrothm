#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/22/20 10:21
# @Author  : Nan Wang
# @Site    : 
# @File    : 200_longest_palindromic_substring.py
# @Software: PyCharm

"""

描述
中文
English
Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.

您在真实的面试中是否遇到过这个题？
样例
Example 1:

Input:"abcdzdcab"
Output:"cdzdc"
Example 2:

Input:"aba"
Output:"aba"
挑战
O(n2) time is acceptable. Can you do it in O(n) time.


"""


class Solution:
    """
    @param s: input string
    @return: the longest palindromic substring
    """

    def longestPalindrome(self, s):
        # write your code here
        ph = ''
        for i in reversed(range(int(len(s) / 2) + 1)):
            if s[i] == s[len(s) - i - 1]:
                ph = s[i] + ph + s[len(s) - i - 1]
            else:
                break
        if len(s) % 2 == 1: # 偶数
            ph = ph[:int(len(s)/2)]+ph[int(len(s)/2)+1:]
        else: # 奇数
            ph = ph[:int(len(s)/2)]+ph[int(len(s)/2)+2:]
        return ph


if __name__ == '__main__':
    s1 = "abb"
    s2 = "a"
    print(Solution().longestPalindrome(s1))
    print(Solution().longestPalindrome(s2))

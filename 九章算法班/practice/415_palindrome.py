#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/22/20 09:32
# @Author  : Nan Wang
# @Site    : 
# @File    : 415huiwenchuan.py
# @Software: PyCharm
"""

Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Example 1:

Input: "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama"
Example 2:

Input: "race a car"
Output: false
Explanation: "raceacar"

挑战
O(n) time without extra memory.

"""

class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        # write your code here
        placeholder = ''
        s = s.lower()
        for p in s:
            if p in '1234567890poiuytrewqasdfghjklmnbvcxz':
                placeholder = placeholder + p
        for i in range(int(len(placeholder)/2)):
            if placeholder[i] != placeholder[len(placeholder)-i-1]:
                return False
        return True

if __name__ == '__main__':
    s1 = "A man, a plan, a canal: Panama"
    s2 = "race a car"
    print(Solution().isPalindrome(s1))
    print(Solution().isPalindrome(s2))














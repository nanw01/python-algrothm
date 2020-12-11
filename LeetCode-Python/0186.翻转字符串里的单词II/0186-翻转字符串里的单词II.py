class Solution(object):
    def reverseWords(self, string):
        """
        :type str: List[str]
        :rtype: None Do not return anything, modify str in-place instead.
        """
        l = len(string)
        if not l:
            return 
        
        def reverse(start, end): #���ߺ����������ǽ�string[start:end + 1]��ת
            left, right = start, end
            while(left < right):
                string[left], string[right] = string[right], string[left]
                left += 1
                right -= 1
        
        reverse(0, l - 1) #��������ת
        first_char_idx = 0
        for i, x in enumerate(string):
            if x == " ":
                reverse(first_char_idx, i - 1) #�ٰ�ÿ�����ʽ��з�ת
                first_char_idx = i + 1

        reverse(first_char_idx, l - 1)#�����һ�����ʷ�ת
class Solution(object):
    def prevPermOpt1(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        l = len(A)
        if sorted(A) == A:
            return A
        
        for i in range(l - 1, -1, -1): #��֤����һ�εݼ�
            if A[i - 1] > A[i]: #��һ��������
                break
        #�ƶ�i - 1
        print A[i - 1]
        for j in range(l - 1, i - 1, -1): #�ұ�A[i-1]С������
            if A[j] < A[i - 1]:
                # j -= 1
                break
        print i, j
        A[i - 1],A[j] = A[j],A[i - 1]
        print A
        return A
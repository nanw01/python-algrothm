class Solution(object):
    def numEnclaves(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        m, n = len(A), len(A[0])
        
        def dfs(x, y):
            dx = [0, 1, -1, 0]
            dy = [-1, 0, 0, + 1]
            for k in range(4):
                xx = x + dx[k]
                yy = y + dy[k]

                if xx >= 0 and xx < m and yy >= 0 and yy < n: #��ǰ������Ч
                    if A[xx][yy] == 1:
                        A[xx][yy] = 2 #���߹��ĵ�Ⱦɫ
                        dfs(xx, yy)
        #--------�����ڴ��������������ڵ����е�
        i = 0
        for j in range(n): #�ϱ�
            if A[i][j] == 1:
                A[i][j] = 2
                dfs(i, j)

        i = m - 1
        for j in range(n): #�±�
            if A[i][j] == 1:
                A[i][j] = 2
                dfs(i, j)    
        j = 0        
        for i in range(m): #���
            if A[i][j] == 1:
                A[i][j] = 2
                dfs(i, j) 

        j = n - 1     
        for i in range(m): #�ұ�
            if A[i][j] == 1:
                A[i][j] = 2
                dfs(i, j) 
        #--------�����ڴ��������������ڵ����е�
        
        res = 0
        for i in range(m):
            for j in range(n):
                if A[i][j] == 1: #ͳ��һ�»��ж��ٸ�û��ȥ���ĵ�
                    res += 1
          
        return res
class Solution(object):
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # �����Ϸ��ߣ�x -= 1, y += 1
        # �����½��ߣ�x += 1, y -= 1
        m = len(matrix)
        if not m :
            return []
        n = len(matrix[0])
        if not n :
            return []

        cnt = 0
        x, y = 0, 0
        res = list()
        direction = "right"
        while(cnt < m * n):
            cnt += 1
            # print direction, x, y, matrix[x][y]
            res.append(matrix[x][y])
            if direction == "right":#�����Ϸ���
                if x >= 1 and y < n - 1:
                    x -= 1
                    y += 1
                    continue
                else:
                    direction = "left" #��������
                    if x == 0 and y < n - 1: #���ϱ�
                        y += 1
                    elif  y == n - 1: #���ұ�
                        x += 1
            else: # �����·���
                if x < m - 1 and y >= 1:
                    x += 1
                    y -= 1
                    continue
                else:
                    direction = "right" #��������
                    if x == m - 1: # ���±�
                        y += 1
                    elif y == 0 and x < m - 1: #�����
                        x += 1 
            # print res
        return res
                        
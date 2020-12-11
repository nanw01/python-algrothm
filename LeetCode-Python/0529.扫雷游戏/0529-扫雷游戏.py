class Solution(object):
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        if not board or not board[0]:
            return board
        
        m, n = len(board), len(board[0])
        visited = [[0 for _ in range(n + 1)] for j in range(m + 1)]
        
        x, y = click[0], click[1]
        if board[x][y] == "M":#һ�¾��ڵ����ף�����ֱ���޸�Ȼ�󷵻�
            board[x][y] = "X"
            return board
        
        dx = [1, 1, -1, -1, 0, 0, -1, 1]
        dy = [1, 0, -1, 0, 1, -1, 1, -1]
        
        def dfs(x0, y0):
            if board[x0][y0] == "M" or visited[x0][y0] == 1: #����õ��ǵ��ף������Ѿ������ˣ���ֱ����·
                return
                                               
            visited[x0][y0] = 1 #���һ��������
            mineCnt = 0         #ͳ�Ƶ�ǰ�㸽���м�������
            for k in range(8):
                x = x0 + dx[k]
                y = y0 + dy[k]
                
                if 0 <= x < m and 0 <= y < n and board[x][y] == "M":
                    mineCnt += 1
            
            if mineCnt > 0:
                board[x0][y0] = str(mineCnt) #������ף���ֱ��˵�м�����
            else:
                board[x0][y0] = 'B' #û�����ڵ��ף��ͻ���Ҫ�ݹ���Χ�ĵ�

                for k in range(8):
                    x = x0 + dx[k]
                    y = y0 + dy[k]

                    if 0 <= x < m and 0 <= y < n and visited[x][y] == 0:
                        dfs(x, y)

        dfs(x, y)
        return board
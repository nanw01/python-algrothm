
#dfs R���ܱ�b��ס��Ŀ���ǳ�p���ĸ������ƶ�
class Solution(object):
    def numRookCaptures(self, board):
        """
        :type board: List[List[str]]
        :rtype: int
        """
        lx = 8
        ly = 8
        res = 0
        for index, item in enumerate(board):
            if "R" in item:
                Rposx, Rposy = index, item.index("R")

        posx = Rposx   
        posy = Rposy     
        while(posx > 0):
            if board[posx][posy] == "B":
                break
            elif board[posx][posy] == "p":
                res += 1
                break
            posx -= 1

        posx = Rposx   
        posy = Rposy
        while(posx < lx):
            
            if board[posx][posy] == "B":
                break
            elif board[posx][posy] == "p":
                res += 1
                break
            posx += 1

        posx = Rposx   
        posy = Rposy
        while(posy >=0):

            if board[posx][posy] == "B":
                break
            elif board[posx][posy] == "p":
                res += 1
                break
            posy -= 1

        posx = Rposx   
        posy = Rposy
        while(posy < ly):
            if board[posx][posy] == "B":
                break
            elif board[posx][posy] == "p":
                res += 1
                break
            posy += 1
        return res
                
        
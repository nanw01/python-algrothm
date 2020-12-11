class Solution(object):
    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #̰��+trierTree
        root = TreeNode(-1)
        
        for num in nums:
            cur_node = root #��ǰ��node
            
            for i in range(0, 32):               #����32��λ
                # print num, 1 <<(31 - i), num & (1 <<(31 - i))
                if num & (1 <<(31 - i)) == 0:    #�����ǰλ������Ľ����1�� ��������
                    if not cur_node.left:
                        cur_node.left = TreeNode(0)
                    cur_node = cur_node.left
                else:                            #�����ǰλ������Ľ����0�� ��������
                    if not cur_node.right:
                        cur_node.right = TreeNode(1)
                    cur_node = cur_node.right
            cur_node.left = TreeNode(num)        #��������Ҷ�ӽڵ��¼һ���������ֵ
                    
        res = 0
        for num in nums:
            cur_node = root
            
            for i in range(0, 32):
                # print cur_node.val, cur_node.left, cur_node.right
                if num & (1 <<(31 - i)) == 0:     #��������Ϊ0������������ߣ��������ߣ���Ϊ����������1����������һλ������õ�1
                    if cur_node.right:           #��������
                        cur_node = cur_node.right#��������
                    else:                        #����������
                        cur_node = cur_node.left#��������
                else:                            #��������Ϊ1������������ߣ��������ߣ���Ϊ����������0����������õ�1
                    if cur_node.left:            #��������
                        cur_node = cur_node.left#��������
                    else:                        #����������
                        cur_node = cur_node.right#��������  
            temp = cur_node.left.val             #�õ�����·����ŵ�����ֵ
                
            res = max(res, num ^ temp)           #ÿ��ˢ��resΪ���ֵ
                
        return res
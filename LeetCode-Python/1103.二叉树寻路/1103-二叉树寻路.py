class Solution(object):
    def pathInZigZagTree(self, label):
        """
        :type label: int
        :rtype: List[int]
        """
        def findLayer(target, root_val, layer): #��ĳ��label���ڵڼ���
            if root_val + 2 ** (layer) < target:
                return findLayer(target, root_val + 2 ** (layer), layer + 1)
            else:
                return layer + 1
        
        layer = findLayer(label, 1, 1)
        res = []
        while label > 1:
            for i in range(2 ** (layer - 1), 2 **(layer)):
                if i == label:#�ڵ�ǰ���ҵ���label
                    idx = i - 2 ** (layer - 1) #label�ڱ�����±�
                    parentidx = (2 ** (layer - 1) - 1 - idx )// 2 #label�ĸ��ڵ�����һ����±�
                    # print cnt, parentcnt
                    break
            res = [label] + res
            label = [i for i in range(2 ** (layer - 2), 2 **(layer - 1))][parentidx]
            layer -= 1
        return [1] + res
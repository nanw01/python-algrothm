class Solution(object):
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        dict = {}
        for i, item in enumerate(list1):
            dict[item] = i
            
        minIdxSum = 2 ** 32
        res = []
        for i, item in enumerate(list2):
            if item in dict: #��ǰ�������߶���
                tmp = i + dict[item]
                if tmp < minIdxSum: #��Ҫ������С������
                    res = [item]
                    minIdxSum = tmp
                elif tmp == minIdxSum: #��Ŀǰ����С������
                    res.append(item)
        return res
            
        
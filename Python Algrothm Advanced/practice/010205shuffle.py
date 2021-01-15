class Solution(object):
    def shuffle(self, array):
        """
        input : int[] array
        """
        from random import randint
        for k in range(len(array)):
            i = randint(0, len(array)-1)
            array[i], array[k] = array[k], array[i]
        return array

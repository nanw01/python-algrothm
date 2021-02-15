
import random


class Solution(object):
    def __init__(self):
        """
        complete the constructor if needed.
        """
        self.currSample = 0
        self.count = 0

    def read(self, value):
        """
        read a value in the stream.
        :type: value: int
        """
        self.count += 1
        rand = random.randint(0, self.count - 1)
        if rand == 0:
            self.currSample = value

    def sample(self):
        """
        return the sample of already read values.
        :rtype: int
        """
        return self.currSample

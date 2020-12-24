class Solution(object):
    def numberOfNodes(self, head):
        """
        input: ListNode head
        return: int
        """
        # write your solution here
         temp = self.head  # Initialise temp
          count = 0  # Initialise count

           # Loop while end of linked list is not reached
           while (temp):
                count += 1
                temp = temp.next
            return count

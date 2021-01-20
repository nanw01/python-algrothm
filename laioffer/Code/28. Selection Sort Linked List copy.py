# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def selectionSort(self, head):
        """
        input: ListNode head
        return: ListNode
        """
        # write your solution here
        new_head = ListNode(None)
        new_head.next = head
        tail = new_head

        while tail.next:
            prev, curr = tail, tail.next
            min_node, min_node_predecessor = curr, prev

            # 找到最小值
            while curr:
                if curr.val < min_node.val:
                    min_node, min_node_predecessor = curr, prev
                prev, curr = curr, curr.next
            # 分离最小值
            min_node_predecessor.next = min_node.next

            # 添加到上一个最小值的后面
            next = tail.next
            tail.next = min_node
            min_node.next = next
            tail = tail.next
        return new_head.next


def printList(head):
    if head is None:
        print(' ')
        return
    curr_node = head
    while curr_node:
        print(curr_node.val, end=" ")
        curr_node = curr_node.next
    print(' ')


if __name__ == "__main__":
    s = Solution()

    node1 = ListNode(1)
    node2 = ListNode(3)
    node3 = ListNode(5)
    node4 = ListNode(7)
    node5 = ListNode(9)
    node6 = ListNode(2)
    node7 = ListNode(4)
    node8 = ListNode(6)

    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = node5
    node5.next = node6
    node6.next = node7
    node7.next = node8

    new = s.selectionSort(node1)

    printList(new)

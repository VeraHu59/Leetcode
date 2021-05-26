
## 707. Design Linked List
# Solution 1: Single Linked List
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class MyLinkedList:
    def __init__(self):
        self.size = 0
        self.head = ListNode(0)  # sentinel node as pseudo-head
        

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        # if index is invalid
        if index < 0 or index >= self.size:
            return -1
        
        curr = self.head
        # index steps needed 
        # to move from sentinel node to wanted index
        for _ in range(index + 1):
            curr = curr.next
        return curr.val
            

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        # make use of the implemented method
        self.addAtIndex(0, val)
        

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        # make use of the implemented method
        self.addAtIndex(self.size, val)
        

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        # If index is greater than the length, 
        # the node will not be inserted.
        if index > self.size:
            return
        
        # [so weird] If index is negative, 
        # the node will be inserted at the head of the list.
        if index < 0:
            index = 0
        
        self.size += 1
        # find predecessor of the node to be added
        pred = self.head
        for _ in range(index):
            pred = pred.next
            
        # node to be added
        to_add = ListNode(val)
        # insertion itself
        to_add.next = pred.next
        pred.next = to_add
        

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        # if the index is invalid, do nothing
        if index < 0 or index >= self.size:
            return
        
        self.size -= 1
        # find predecessor of the node to be deleted
        pred = self.head
        for _ in range(index):
            pred = pred.next
            
        # delete pred.next 
        pred.next = pred.next.

# Solution 2: Double linked list
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None

class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        self.head = ListNode(0)
        self.tail = ListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def printLinkedList(self):
        curr = self.head
        l = []
        for _ in range(self.size):
            curr = curr.next
            l.append(curr.val)
        print(l)

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index >= self.size or index < 0:
            return -1
        # if the node is closer to the tail
        if self.size - index - 1 < index:
            curr = self.tail
            for _ in range(self.size-index):
                curr = curr.prev

        # if the node is closer to the head
        else:
            curr = self.head 
            for _ in range(index+1):
                curr = curr.next
        # self.printLinkedList()
        return curr.val

        

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        pred, succ = self.head, self.head.next
        
        self.size += 1
      
        added = ListNode(val)
        added.next = succ 
        added.prev = pred 

        pred.next = added
        succ.prev = added
        # self.printLinkedList()


        

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        pred, succ = self.tail.prev, self.tail 

        self.size += 1

        added = ListNode(val)
        added.next = succ
        added.prev = pred
        pred.next = added
        succ.prev = added  
        # self.printLinkedList()
        

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        # check the edge cases
        if index > self.size:
            return 
        elif index < 0:
            index = 0
        else:
            if self.size - index - 1 < index:
                curr = self.tail
                for _ in range(self.size-index):
                    curr = curr.prev
                
            # if the node is closer to the head
            else:
                curr = self.head 
                for _ in range(index+1):
                    curr = curr.next
            pred = curr.prev 
            added = ListNode(val)
            added.next = curr
            added.prev = pred
            pred.next = added
            curr.prev = added
            self.size += 1
        # self.printLinkedList()

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index > self.size - 1 or index < 0:
            return 

        # if the node is closer to the tail
        if self.size - index - 1 < index:
                curr = self.tail
                for _ in range(self.size-index):
                    curr = curr.prev
        # if the node is closer to the head
        else:
            curr = self.head 
            for _ in range(index+1):
                curr = curr.next

        pred = curr.prev
        pred.next = curr.next 
        curr.next.prev = pred 
        self.size -= 1
        # self.printLinkedList()

## 141. Linked List Cycle

# Solution 1: Hash Table
# use a hash set to store all the nodes we have seen 
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        seen = set()
        
        while head is not None:
            if head in seen:
                return True
            
            seen.add(head)
            head = head.next
       
        return False


# Solution 2: Two pointers
# Time complexity is O(N+K), where k is the cyclic length
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if head is None:
            return False
        slow = fast = head
        
        # the final status if there is no cycle
        while fast is not None and fast.next is not None:
            slow = slow.next 
            fast = fast.next.next
            if slow == fast:
                return True
        
        return False

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
    	# edge case
        if head is None:
            return False
        # you should ensure that at the beginning, fast != slow
        slow = head
        fast = head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True

## 142. Linked List Cycle II

# Solution 1: Hash Table

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        seen = set()
        while head is not None:
            if head in seen:
                return head
            seen.add(head)
            head = head.next
            
        return None

# Solution 2: Two pointers

class Solution:
    def detectIntersection(self, head):
        slow = fast = head
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next 
            if fast == slow:
                return slow
        return None
    
    def detectCycle(self, head: ListNode) -> ListNode:
        # edge case
        if head is None:
            return None
        
        # detect the intersection
        intersection = self.detectIntersection(head)
        # if there is no cycle in the linked list
        if intersection is None:
            return None
        
        # find the start of cycle
        while intersection != head:
            intersection = intersection.next
            head = head.next
        
        return intersection


## 160. Intersection of Two Linked Lists

# Solution 1: Hash Table 
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # traverse the first linked 
        seen = set()
        while headA is not None:
            seen.add(headA)
            headA = headA.next
        # whether there is visited node in the second linked list    
        while headB is not None:
            if headB in seen:
                return headB
            headB = headB.next
        
        return None

# Solution 2: Two pointers
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        pA = headA
        pB = headB

        while pA != pB:
            pA = headB if pA is None else pA.next
            pB = headA if pB is None else pB.next

        return pA


## 19. Remove Nth Node From End of List

# Solution 1: Two passes
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # edge case
        if head is None:
            return None
        # find the length of the linked list
        curr = head
        length = 0
        while curr is not None:
            length += 1
            curr = curr.next
        # delete the Nth from the end
        if n > length:
            return head
        elif n == length:
            return head.next
        temp = head
        for _ in range(length-n-1):
            temp = temp.next
        temp.next = temp.next.next
        
        return head

# Solution 2: One pass
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if head is None:
            return None
        
        fast = slow = head

        while n >= 0:
            if fast is None and n > 0:
                return head
            elif fast is None and n == 0:
                return head.next
            fast = fast.next
            n -= 1
  
        while fast is not None:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        
        return head

## 206. Reverse Linked List

# Solution 1: Iterative
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        
        while head is not None:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev


# Solution 2: Recursion

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head is None:
            return None
        elif head.next is None:
            return head
        else:
            # store the next node
            next_node = head.next
            # point the head to None
            head.next = None
            # reverse the rest of the list
            reversed_next = self.reverseList(next_node)
            # connect the rest of the list back to the head
            next_node.next = head
            return reversed_next






## 203. Remove Linked List Elements

class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        sentinel = ListNode(0)
        sentinel.next = head
        
        prev, curr = sentinel, head
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        
        return sentinel.next


## 328. Odd Even Linked List

class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
    	# you must be very careful about the edge case of linked list
        if head is None or head.next is None:
            return head
        # Do not directly use head and even_head, since you need them later
        curr = head
        even_head = head.next
        even = even_head
        while even is not None and even.next is not None:
            curr.next = even.next
            # update the curr!
            curr = curr.next
            even.next = curr.next
            # update the even
            even = even.next
        curr.next = even_head
        
        return head


## 234. Palindrome Linked List

# Solution 1: Two pointers
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # traverse the linked list and copy all values into list
        values = []
        while head is not None:
            values.append(head.val)
            head = head.next
        
        # use two pointers method to check whether the list is a palindrome or not 
        left = 0
        right = len(values)-1
        
        while left <= right:
            if values[left] != values[right]:
                return False
            left += 1
            right -= 1
        
        return True


## 21. Merge Two Sorted Lists

# Solution 1: Iteration
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        sentinel = ListNode(0)
        curr = sentinel
        while l1 is not None and l2 is not None:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next 
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
            
        if l2 is None:
            curr.next = l1
        elif l1 is None:
            curr.next = l2
        else:
            pass
        
        return sentinel.next

# Solution 2: Recursion

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    	# exit condition
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        else:
        	# statement
            if l1.val <= l2.val:
                l1.next = self.mergeTwoLists(l1.next, l2)
                return l1
            else: 
                l2.next = self.mergeTwoLists(l1, l2.next)
                return l2


## 2. Add Two Numbers

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # edge case
        if l1 is None and l2 is None:
            return None
        elif l1 is None:
            return l2
        elif l2 is None:
            return l1
        
        sentinel = ListNode(0)
        curr = sentinel
        add_one = 0
        
        while l1 is not None or l2 is not None:
            if l1 is not None and l2 is not None:
                curr_val = l1.val + l2.val + add_one
                l1 = l1.next
                l2 = l2.next
            elif l1 is None:
                curr_val = l2.val + add_one
                l2 = l2.next
            else:
                curr_val = l1.val + add_one
                l1 = l1.next
            
            if curr_val >=10:
                add_one = 1
                curr_val -= 10
            else:
                add_one = 0
            curr.next = ListNode(curr_val)
            curr = curr.next
        
        if add_one == 1:
            curr.next = ListNode(1)

        
        
        return sentinel.next


# A more elegant solution 
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = ListNode(0)
        result_tail = result
        carry = 0
                
        while l1 or l2 or carry:            
            val1  = (l1.val if l1 else 0)
            val2  = (l2.val if l2 else 0)
            carry, out = divmod(val1+val2 + carry, 10)    
                      
            result_tail.next = ListNode(out)
            result_tail = result_tail.next                      
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
               



## 430. Flatten a Multilevel Doubly Linked List
# Solution 1: DFS by recursion
class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        # edge case
        if not head:
            return head
        sentinel = Node(None, None, head, None)
        head.prev = sentinel
        
        self.flatten_dfs(sentinel, head)
        # detach the head from the sentinel node
        sentinel.next.prev = None
        
        return sentinel.next
        
    def flatten_dfs(self, prev, curr):
    	# exit condition
        if curr is None:
            return prev
        
        curr.prev = prev
        prev.next = curr
        
        tempNext = curr.next
        # recurse on the left subtree
        tail = self.flatten_dfs(curr, curr.child)
        curr.child = None
        # recurse on the right subtree
        return self.flatten_dfs(tail, tempNext)


# Solution 2: DFS by iteration
# Make use of Stack. Last In First Out

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""
class Solution(object):
    def flatten(self, head):
        if not head:
            return

        pseudoHead = Node(0,None,head,None)
        prev = pseudoHead

        stack = []
        stack.append(head)

        while stack:
            curr = stack.pop()

            prev.next = curr
            curr.prev = prev

            if curr.next:
                stack.append(curr.next)
 
            if curr.child:
                stack.append(curr.child)
                # don't forget to remove all child pointers.
                curr.child = None

            prev = curr
        # detach the pseudo head node from the result.
        pseudoHead.next.prev = None
        return pseudoHead.next


## 138. Copy List with Random Pointer
# Solution 1: Iteration
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # edge case
        if not head:
            return head
       
        # replicate each node.val and node.next first 
        curr = head
        
        while curr:
            copy = Node(curr.val, curr.next, None)
            curr.next = copy
            curr = copy.next
            
        # replicate node.random
        curr = head
        while curr and curr.next:
            curr.next.random = curr.random.next if curr.random else None
            curr = curr.next.next
        
        # detach the copied linked list from the original list
        new_head = head.next
        old_list = head
        new_list = new_head
        while old_list and old_list.next:
            old_list.next = old_list.next.next
            new_list.next = new_list.next.next if new_list.next else None
            old_list = old_list.next
            new_list = new_list.next
        return new_head

# Solution 2: Recursion
class Solution(object):
    """
    :type head: Node
    :rtype: Node
    """
    def __init__(self):
        # Dictionary which holds old nodes as keys and new nodes as its values.
        self.visitedHash = {}

    def copyRandomList(self, head):

        if head == None:
            return None

        # If we have already processed the current node, then we simply return the cloned version of it.
        if head in self.visitedHash:
            return self.visitedHash[head]

        # create a new node
        # with the value same as old node.
        node = Node(head.val, None, None)

        # Save this value in the hash map. This is needed since there might be
        # loops during traversal due to randomness of random pointers and this would help us avoid them.
        self.visitedHash[head] = node

        # Recursively copy the remaining linked list starting once from the next pointer and then from the random pointer.
        # Thus we have two independent recursive calls.
        # Finally we update the next and random pointers for the new node created.
        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)

        return node

class Solution(object):
    def __init__(self):
        # Creating a visited dictionary to hold old node reference as "key" and new node reference as the "value"
        self.visited = {}

    def getClonedNode(self, node):
        # If node exists then
        if node:
            # Check if its in the visited dictionary          
            if node in self.visited:
                # If its in the visited dictionary then return the new node reference from the dictionary
                return self.visited[node]
            else:
                # Otherwise create a new node, save the reference in the visited dictionary and return it.
                self.visited[node] = Node(node.val, None, None)
                return self.visited[node]
        return None

    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """

        if not head:
            return head

        old_node = head
        # Creating the new head node.       
        new_node = Node(old_node.val, None, None)
        self.visited[old_node] = new_node

        # Iterate on the linked list until all nodes are cloned.
        while old_node != None:

            # Get the clones of the nodes referenced by random and next pointers.
            new_node.random = self.getClonedNode(old_node.random)
            new_node.next = self.getClonedNode(old_node.next)

            # Move one step ahead in the linked list.
            old_node = old_node.next
            new_node = new_node.next

        return self.visited[head]


## 61. Rotate List

# My Solution
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
        # get the length of the list
        curr = head
        length = 0
        while curr:
            curr = curr.next
            length += 1
        
        # update k
        k = k % length
        if k == 0:
            return head
        
        # find the kth node from the end
        # Optimization: since we already know the length of the linked list, we can directly find the length-k node, rather than use two pointers
        slow = fast = head
        
        for _ in range(k):
            fast = fast.next
        
        while fast.next:
            slow = slow.next
            fast = fast.next
        
        # swap sub-lists
        fast.next = head
        new_head = slow.next
        slow.next = None
        
        return new_head

# Optimized Solution
class Solution:
    def rotateRight(self, head: 'ListNode', k: 'int') -> 'ListNode':
        # base cases
        if not head:
            return None
        if not head.next:
            return head
        
        # close the linked list into the ring
        old_tail = head
        n = 1
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        old_tail.next = head
        
        # find new tail : (n - k % n - 1)th node
        # and new head : (n - k % n)th node
        new_tail = head
        for i in range(n - k % n - 1):
            new_tail = new_tail.next
        new_head = new_tail.next
        
        # break the ring
        new_tail.next = None
        
        return new_head
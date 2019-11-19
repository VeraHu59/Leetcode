# Questions

Here is the list of questions which use the knowledge of linked list.

- [138 Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)\
- [160 Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)
- [206 Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
- [237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)





## 138. Copy List with Random Pointer (Median)

- **Link:** https://leetcode.com/problems/copy-list-with-random-pointer/

### Solution 1: Iteration
> **Explanation**
>> This idea of this solution can be broken down into three steps:
>> - Create a duplicate of each node just follows the original node
>> - Iterate the new list and assign random pointer for each node
>> - Restore the original list and extract the new linked list
>
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(1).

```python
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        # Iterate the original list and duplicate each node. 
        # The duplicate of each node follows its original immediately.
        if not head:
            return None
        tmp = head
        while tmp:
            new_node = Node(tmp.val, tmp.next, tmp.random)
            tmp.next = new_node
            tmp = tmp.next.next
        
        # Iterate the new list and assign the random pointer for each duplicated node.
        tmp = head
        while tmp:
            if tmp.random:
                tmp.next.random = tmp.random.next
            tmp = tmp.next.next
        
        # Restore the original list and extract the duplicated nodes.
        new_head = head.next
        pold = head
        pnew = new_head
        while pnew.next:
            pold.next = pnew.next
            pold = pold.next
            pnew.next = pold.next
            pnew = pnew.next
        pnew.next = None
        pold.next = None
        return new_head
```

### Solution 2: Recursion
> **Explanation**
>> All we do in this approach is to just traverse the graph and clone it. Cloning essentially means creating a new node for every unseen node you encounter. The traversal part will happen recursively in a depth first manner. Note that we have to keep track of nodes already processed because, as pointed out earlier, we can have cycles because of the random pointers.(from Leetcode Solution)
>
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N).

```python
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
```


## 160. Intersection of Two Linked Lists (Easy)

- **Link:** https://leetcode.com/problems/intersection-of-two-linked-lists/

### Solution 1: Hash set
> **Explanation**
>>  We can store list A in the hash set and iterate through list B. If we find node in B has already appears in the hash set, then this node is the intersection between two lists.
>
> **Complexity Analysis**  
>> - Time complexity : O(N+M), where N and M are the length of list A, B. 
>> - Space complexity : O(M) or O(N).

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pa, pb = headA, headB
        
        if pa is None or pb is None:
            return None
        h = set()
        
        while(pa is not None):
            h.add(pa)
            pa = pa.next
        
        while(pb is not None):
            if pb in h:
                return pb
            else:
                pb = pb.next
        
        return None
```


### Solution 2: Two Pointers
> **Explanation**
>>  We can optimize the above method by using two pointers. In this way, we don't have to use extra space to store hash set.
>
> **Complexity Analysis**  
>> - Time complexity : O(N+M), where N and M are the length of list A, B. 
>> - Space complexity : O(1).

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pa, pb = headA, headB
        
        if pa is None or pb is None:
            return None
        
        while (pa != pb):
            pa = headB if pa is None else pa.next 
            pb = headA if pb is None else pb.next
        
        return pa


```

## 206. Reverse Linked List (Easy)

- **Link:** https://leetcode.com/problems/reverse-linked-list/

### Solution 1: Iteration
> **Explanation**
>>While you are traversing the list, change the current node's next pointer to point to its previous element. Since a node does not have reference to its previous node, you must store its previous element beforehand. You also need another pointer to store the next node before changing the reference. Do not forget to return the new head reference at the end!(From Leetcode solution)
>
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(1).

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        return prev
```

### Solution 2: Recursion
> **Explanation**
>> Assume from node n_k+1 to n_m had been reversed and you are at node nk.
>>
>> n1 → … → nk-1 → nk → nk+1 ← … ← nm
>>
>> So what we want is We want nk+1’s next node to point to nk.(From Leetcode solution)
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N).

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # Bash case: when head or head.next is none, we start to return
        if not head or not head.next:
            return head
        
        p = self.reverseList(head.next)
        head.next.next = head
        # if we don't add this line, your linked list has a cycle in it. 
        # This bug could be caught if you test your code with a linked list of size 2.
        head.next = None
        
        return p
```
## 234. Palindrome Linked List (Easy)

- **Link:** https://leetcode.com/problems/palindrome-linked-list/

### Solution

> **Explanation**
>> We can use two pointers to find the midpoint of the linked list. Then we utilize the LIFO property of stack to store the right side of the list and compare the right side with the left side.

> **Complexity Analysis**  
>> - Time complexity : O(3N) = O(N). 
>> - Space complexity : O(1).

```python
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """   
              
        if not head or not head.next:
            return True
        fast = slow = curr = head
        # find mid point
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        #slow is the midpoint
        
        # push the element of right side into stack
        stack = [slow.val]
        while slow.next:
            slow = slow.next
            stack.append(slow.val)
            
        # compare the right side with the left side
        while stack:
            if curr.val != stack.pop():
                return False
            curr = curr.next
        
        return True
```

## 237. Delete Node in a Linked List (Easy)

- **Link:** https://leetcode.com/problems/delete-node-in-a-linked-list/

### Solution 1: Iteration

> **Complexity Analysis**  
>> - Time complexity : O(1). 
>> - Space complexity : O(1).

```python
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```

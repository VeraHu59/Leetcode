# Questions

Here is the list of questions which use the knowledge of linked list.

- [138 Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)




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

### Solution 2: Recursive
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

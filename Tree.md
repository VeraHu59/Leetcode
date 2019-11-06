# Questions

Here is the list of questions which use the knowledge of Tree.

- [101 Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- [108 Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [104 Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [230 Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [116 Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

# Solutions

## 101. Symmetric Tree (Easy)

- **Link:** https://leetcode.com/problems/symmetric-tree/  


### Solution 1: Recursive
> **Explanation**
>> Two trees are a mirror reflection of each other, if :  
>> 1. Their two roots have the same value.
>> 2. The right subtree of each tree is a mirror reflection of the left subtree of the other tree. 
>
> **Complexity Analysis**  
>> - Time complexity : O(n). Because we traverse the entire input tree once.
>> - Space complexity : O(n).The number of recursive calls is bound by the height of the tree. In the worst case, the tree is linear and the height is in O(n). 


```python
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.isMirror(root.left, root.right)
    
    def isMirror(self, t1, t2):
        if not t1 and not t2:
            return True
        elif not t1 or not t2:
            return False
        else:
            return self.isMirror(t1.left, t2.right) and self.isMirror(t1.right, t2.left) and t1.val == t2.val
```

### Solution 2: Iterative
> **Explanation**
>> We can use stack to do the above steps iteratively.

> **Complexity Analysis**  
>> - Time complexity : O(n)
>> - Space complexity : O(n)


```python
class Solution(object):
    def isSymmetric(self, root):
        if not root:
            return True
        # initialize a stack
        stack = [(root.left, root.right)]
        while stack:
            left, right = stack.pop()
            if not left and not right: # leaf node
                continue
            elif not left or not right:
                return False
            
            if left.val == right.val:
                stack.append((left.left, right.right))
                stack.append((left.right, right.left))
            else:
                return False
        # if it jumps out of the loop, it is true
        return True 
```

## 230. Kth Smallest Element in a BST (Median)

- **Link:** https://leetcode.com/problems/kth-smallest-element-in-a-bst/  


### Solution 1: Depth-First Search (Recursive)
> **Explanation**
>> As this is a Binary Search Tree, the result of inorder traversal is an array sorted in the ascending order. 
>
> **Complexity Analysis**  
>> - Time complexity : O(n).
>> - Space complexity : O(n).

```python
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        def traverse(root):
            if not root:
                return None
            res = []
            
            if root.left:
                res += traverse(root.left)
            res.append(root.val)
            if root.right:
                res += traverse(root.right)
            return res
        
        l = traverse(root)

        return l[k-1]
             
```
> A more pythonic code for this solution (given by Leetcode Solution):
```python
class Solution:
    def kthSmallest(self, root, k):
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
    
        return inorder(root)[k - 1]
```

### Solution 2: Depth-First Search (Iterative)
> **Explanation**
>> Because we only want the kth smallest element, we don't need to build the entire inorder traversal. We could stop iteration after we find the kth element. 
>
> **Complexity Analysis**  
>> - Time complexity : O(H+k), where H is the height of the tree. If this is a balanced tree, then the height is log(N). And for completedly unbalanced tree with all the nodes in the left subtree, the height is N.
>> - Space complexity : O(H+k).
```python
class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            # when k == 0, return the value of node
            if not k:
                return root.val
            root = root.right
``` 

## 116. Populating Next Right Pointers in Each Node (Median)

- **Link:** https://leetcode.com/problems/populating-next-right-pointers-in-each-node/ 

### Solution 1: Breadth-First Search (Queue)
> **Explanation**
>> When solving BFS problems, We usually use queue to store the elements of each level.

> **Complexity Analysis**  
>> - Time complexity : O(n).
>> - Space complexity : O(n). 

```python
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        # the corner case
        if not root:
            return None
        
        queue = [root]
        while queue:
            node = queue.pop(0)
            # ensure that node is not in the last level
            if node.left and node.right:
                node.left.next = node.right
                if node.next:
                    node.right.next = node.next.left
                queue.append(node.left)
                queue.append(node.right)
        return root
```

### Solution 2: Breadth-First Search (Two pointers)
> **Explanation**
>> Since we are manipulating tree nodes on the same level, it's easy to come up with
a very standard BFS solution using queue. But because of next pointer, we actually
don't need a queue to store the order of tree nodes at each level, we just use a next
pointer like it's a link list at each level; In addition, we can borrow the idea used in
the Binary Tree level order traversal problem, which use cur and next pointer to store
first node at each level; we exchange cur and next every time when cur is the last node
at each level (from [tyr034](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/discuss/37465/Python-Solution-With-Explaintion) ).
>
> **Complexity Analysis**  
>> - Time complexity : O(n).
>> - Space complexity : O(1). Since we don't have to store the order of tree nodes.

```python
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        # the corner case
        if not root:
            return None
        
        cur = root
        next = root.left
        
        while next:
            # first link the left and right child of the cur
            cur.left.next = cur.right
            # cur is the last element in the level
            if not cur.next:
                cur = next
                next = cur.left
            # link the right child of cur and the left child of cur.next
            else:
                cur.right.next = cur.next.left
                cur = cur.next
        return root           
```

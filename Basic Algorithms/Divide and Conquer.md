# Binary Tree

**Key Point** 
When you meet Binary Tree questions, the first thing you should do is figuring out the relationship with the result of the whole tree and that of the left child tree.
 
In the best case, the height of binary tree is log(N), where N is the number of nodes. And in the worst case, the height is N.
Therefore, the binary tree is not suitable for recursion, for it may cause stackoverflow.


**Time Complexity**
1. If T(N) = 2T(N/2) + 0(N), the O(N + NlogN)=O(NlogN). The merge sort satiesfy this time complexity. And in the best case, quick sort also satiesfy this condition. If we use binary tree to understand this time complexity,
we will find that in each level of binary tree, the total time we use is O(N). And there are logN levels. Therefore, the overall
time complexity = O(#levels * time spend on each level) = O(NlogN)
2. If T(N) = 2T(N/2) + 0(1), the n*T(1) + O(1+2+4+...+N)=O(N). 

**Binary Search Tree**

The in-order traverse is in ascending order, if no equal value.

 
 
 **Corner cases**
 1. The Binary Tree is null
 2. The root node is also leaf node

## Sample Questions

### 144. Binary Tree Preorder Traversal
**Lind**: https://leetcode.com/problems/binary-tree-preorder-traversal/

1. Recursive

The following two methods are all recursion methods. But the first one return in parameters, but the second one has return value.
- Traverse
```python 
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        self.helper(root, result)
        return result
    
    def helper(self, root, result):
        if not root:
            return 
        result.append(root.val)
        self.helper(root.left, result)
        self.helper(root.right, result)
```

- Divide and Conquer

```python
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        leftarr = self.preorderTraversal(root.left)
        rightarr = self.preorderTraversal(root.right)
        
        return [root.val]+leftarr+rightarr
```

2. Nonrecursive (iterative)

```python
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        
        stack, output = [root], []
        
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
                if root.right is not None:
                    stack.append(root.right)
                if root.left is not None:
                    stack.append(root.left)
        
        return output
```

### 94. Binary Tree Inorder Traversal
**Link**:https://leetcode.com/problems/binary-tree-inorder-traversal/

Iteration

```python
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
             
        curr = root
        stack = []
        res = []
        while stack or curr is not None:
            while curr:
                stack.append(curr)
                curr = curr.left
                
            curr = stack.pop()
            res.append(curr.val)
            curr = curr.right
        return res
```

Recursion

```python
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        self.helper(root, result)
        return result
             
    def helper(self, root, result):
        if root:
            self.helper(root.left, result)
            result.append(root.val)
            self.helper(root.right, result)
```

### 257. Binary Tree Paths
**Lind**: https://leetcode.com/problems/binary-tree-paths/

```python
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        paths = []
        # find all root-to-leaf paths start with path
        def helper(root, path):
            if root:
                path += str(root.val)
                # already arrive at leaf node
                if not root.left and not root.right:
                    paths.append(path)
                else:
                    path += '->'
                helper(root.left, path)
                helper(root.right, path)
                
        helper(root, '')
        return paths
```

### Lint596. Minimum Subtree
**Link**: https://www.lintcode.com/problem/minimum-subtree/description

This is not a pure divide and conquer solution, for it uses class variable `minSum`.
```python
class Solution:
    minSum = float('inf')
    
	   def findSubtree(self, root):
		      minRoot = root
		      self.getSum(root)

		      return minRoot

	   def getSum(self,node):
		      if not node:
			         return 0
		      left = self.getSum(node.left)
		      right = self.getSum(node.right)
		      cur_sum = node.val + left + right
        
		      if cur_sum < self.minSum:
			         self.minSum = cur_sum
			         minRoot = node
            
		      return cur_sum
```

### 236. Lowest Common Ancestor of a Binary Tree

```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        stack = [root]
        parent = {root:None}
        # iterate until we find the parent of both p and q
        while p not in parent or q not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
        
        # find the all ancestors of p
        ancestor = set()
        while p:
            ancestor.add(p)
            p = parent[p]
        
        while q not in ancestor:
            q = parent[q]
            
        return q
```

```python
class Solution:
    """
    @param root: The root of the binary search tree.
    @param A and B: two nodes in a Binary.
    @return: Return the lowest common ancestor(LCA) of the two nodes.
    """ 
    def lowestCommonAncestor(self, root, A, B):
        # A,B 都在 root 为根的二叉树里, return lca(A,B)
        # 如果 A,B 都不在 root 为根的二叉树里, return None
        # 如果只有 A 在，return A
        # 如果只有 B 在，return B
        if root is None:
            return None
            
        if root is A or root is B:
            return root
            
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        
        if left is not None and right is not None:
            return root
        if left is not None:
            return left
        if right is not None:
            return right
        return None
```


### 98. Validate Binary Search Tree
**Link**: https://leetcode.com/problems/validate-binary-search-tree/

If we want to solve this question using Divide & Conquer, we should first of all think over the result of the whole tree and left/right subtree. Then we can easily discover that the whole tree is a validate binary search tree if and only if:
1. The right and left subtree are all validate BST.
2. The maximum value in the left subtree is less than the value of root.
3. The minimum value in the left subtree is larger than the value of root.

Therefore, when we do recursion, we need to record the minimum value and the maximum value.

```python
class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        ifBST, min, max = self.helper(root)
        return ifBST
    def helper(self, root):
        if not root:
            return True, None, None
        
        ifLeftBST, leftMin, leftMax = self.helper(root.left)
        ifRightBST, rightMin, rightMax = self.helper(root.right)
        
        if not ifLeftBST or not ifRightBST:
            return False, None, None
        if root.val <= leftMax and leftMax is not None:
            return False, None, None
        if root.val >= rightMin and rightMin is not None:
            return False, None, None
        
        # if this is a BST, update minNode and maxNode
        # minNode is the minimum value in the left subtree
        # maxNode is the maximum value in the right subtree
        minNode = leftMin if leftMin is not None else root.val
        maxNode = rightMax if rightMax is not None else root.val
        
        return True, minNode, maxNode
```



```python
class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def helper(node, lower = float('-inf'), upper = float('inf')):
            if not node:
                return True
            
            val = node.val
            if val <= lower or val >= upper:
                return False

            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, lower, val):
                return False
            return True

        return helper(root)
```
### 173. Binary Search Tree Iterator
**Link**: https://leetcode.com/problems/binary-search-tree-iterator/
```python
class BSTIterator(object):

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.index = -1
        self.sorted_nodes = []
        self.inorder(root)
        
    def inorder(self, root):
        if not root:
            return 
        self.inorder(root.left)
        self.sorted_nodes.append(root.val)
        self.inorder(root.right)
        

    def next(self):
        """
        @return the next smallest number
        :rtype: int
        """
        self.index += 1
        return self.sorted_nodes[self.index]
        
        

    def hasNext(self):
        """
        @return whether we have a next smallest number
        :rtype: bool
        """
        return self.index + 1 < len(self.sorted_nodes)
``` 

### 109. Convert Sorted List to Binary Search Tree
**Link**: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/

As T(N) = 2T(N/2)+O(N), the final time complexity of this method is O(NlogN).

```python
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        
        mid = self.findMiddle(head)

        root = TreeNode(mid.val)
        
        if mid == head:
            return root
        
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(mid.next)
        return root
        
    # find the middle of the linked list
    # disconnect the right half and the left half
    def findMiddle(self, head):
        slow = fast = head
        prev = None
        
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        # disconnect
        if prev:
            prev.next = None
        return slow        
```

The simple method to optimize the above solution is trasferring the linked list into list and select the middle node by picking up the (left + right)/2 element in the list.

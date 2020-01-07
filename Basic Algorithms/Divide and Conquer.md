# Binary Tree

**Key Point**: When you meet Binary Tree questions, the first thing you should do is figuring out the relationship with the result of the whole tree and
that of the left child tree.
 
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



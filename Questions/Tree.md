# Questions

Here is the list of questions which use the knowledge of Tree.

- [101 Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- [108 Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [104 Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [230 Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [116 Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
- [103 Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- [236 Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [98 Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [105 Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [102 Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [94 Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

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
>> - Time complexity : O(N). Because we traverse the entire input tree once.
>> - Space complexity : O(N).The number of recursive calls is bound by the height of the tree. In the worst case, the tree is linear and the height is in O(N). 


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
>> - Time complexity : O(N)
>> - Space complexity : O(N)


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
## 108. Convert Sorted Array to Binary Search Tree (Easy)
- **Link:** https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/ 

### Solution: Recursion
> **Explanation**
>> We can first the root of the tree, which is the middle element in the array. Then, we build the left child and right child recursively.
>
> **Complexity Analysis**  
>> - Time complexity : O(N). As time complexity T(n) satisfies the recurrence T(n) = 2T(n/2) + O(1), which solves to T(n) = O(n).
>> - Space complexity : O(H), where H is the height of BST. The only extra space we use is storing mid - O(1), so the space we use in each level is constant.

```python
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def convert(left, right):
            if left > right:
                return None
            # find the root
            mid = (left+right)//2
            node = TreeNode(nums[mid])
            # build tree recursively
            node.left = convert(left, mid-1)
            node.right = convert(mid+1, right)
            return node
        
        
        return convert(0,len(nums)-1)
                 
```

## 104. Maximum Depth of Binary Tree (Easy)
- **Link:** https://leetcode.com/problems/maximum-depth-of-binary-tree/

### Solution 1: Recursion
> **Explanation**
>> If the root is not null, intuitivly, the height of binary tree equals to one plus the longer distance of its subtree.
>
> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return 1+max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0               
```
### Solution 2: Breadth-First-Search
> **Explanation**
>> We can do BFS using queue.

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        level = [root] if root else[]
        depth = 0
        while level:
            depth += 1
            # push the child nodes of every node in level into the queue
            queue = []
            for node in level:
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            level = queue
        # if there is no child node, then we jump out of loop and return depth
        return depth
```
> We can also do BFS using stack.

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        depth=0
        stack = [(root,1)]
    
    
        while stack:
            root,leng=stack.pop()
            if not root:
                return 0
        
            if leng > depth:
                depth = leng            
        
            if root.right:
                stack.append((root.right,leng+1))
            if root.left:
                stack.append((root.left,leng+1)) 
            
        return depth
```

## 230. Kth Smallest Element in a BST (Median)

- **Link:** https://leetcode.com/problems/kth-smallest-element-in-a-bst/  


### Solution 1: Depth-First-Search (Recursive)
> **Explanation**
>> As this is a Binary Search Tree, the result of inorder traversal is an array sorted in the ascending order. 
>
> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

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

### Solution 2: Depth-First-Search (Iterative)
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

### Solution 1: Breadth-First-Search (Queue)
> **Explanation**
>> When solving BFS problems, We usually use queue to store the elements of each level.

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N). 

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

### Solution 2: Breadth-First-Search (Two pointers)
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
>> - Time complexity : O(N).
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
## 103. Binary Tree Zigzag Level Order Traversal (Median)

- **Link:** https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/

### Solution: Level Traversal (Queue)
> **Explanation**
>> We just need to do simple level traversal. And using depth to decide whether from left to right or from right to left. If the depth is odd, we reverse the list of that level. Otherwise, we return the original list.

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N). 

```python
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return None
        
        queue, res = [root], []
        depth = 0
        
        while queue:
            level = []
            size = len(queue)
            for i in range(size):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if depth % 2 == 0:
                res += [level]
            else:
                level.reverse()
                res += [level]
            depth+=1
                
        return res
```
## 236. Lowest Common Ancestor of a Binary Tree (Median)

- **Link:** https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/


### Solution 1: Depth-First-Search (Backtrack)
> **Explanation**
>> **Algorithm**
>>> 1. Start traversing the tree from the root node.
>>> 2. If the current node itself is one of p or q, we would mark a variable mid as True and continue the search for the other node in the left and right branches.
>>> 3. If either of the left or the right branch returns True, this means one of the two nodes was found below.
>>> 4. If at any point in the traversal, any two of the three flags left, right or mid become True, this means we have found the lowest common ancestor for the nodes p and q.(from leetcode solution)

> **Complexity Analysis**  
>> - Time complexity : O(N), where N is the number of nodes in the binary tree. 
>> - Space complexity : O(N). For each recursion, we need constant space. But the times we call recursion function depends on the height of given tree. If this is a completely unbalanced tree, the height is N. 


```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def traverse_tree(curr):
            # if reached the end of a branch, return false
            if not curr:
                return False
            # whether the current node equal to p or q
            mid = curr == p or curr == q
            # check if there is p or q in the left or right child. If true, return 1; else return 0
            left = traverse_tree(curr.left)
            right = traverse_tree(curr.right)
            
            if mid+left+right>=2:
                self.ans = curr
            # if any of the mid, left, right is true, this path has the node we search
            return mid or left or right
        
        traverse_tree(root)
        return self.ans
```

### Solution 2: Iterative with parent pointers
> **Explanation**
>> **Algorithm**
>>> 1. Start from the root node and traverse the tree.
>>> 2. Until we find p and q both, keep storing the parent pointers in a dictionary.
>>> 3. Once we have found both p and q, we get all the ancestors for p using the parent dictionary and add to a set called ancestors.
>>> 4. Similarly, we traverse through ancestors for node q. If the ancestor is present in the ancestors set for p, this means this is the first ancestor common between p and q (while traversing upwards) and hence this is the LCA node.(from leetcode solution)

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N). 

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
 ## 98. Validate Binary Search Tree (Median)

- **Link:** https://leetcode.com/problems/validate-binary-search-tree/


### Solution 1: Recursion
> **Explanation**
>> Note that not only the right child should be larger than the node but all the elements in the right subtree. Therefore, we should keep both upper and lower limits for each node while traversing the tree, and compare the node value not with children values but with these limits. (referring to leetcode solution)

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N). 

```python
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def helper(node, lower = float('-inf'), upper = float('inf')):
            # if the node is null, return true
            if not node:
                return True
            
            if node.val <= lower or node.val >= upper:
                return False
            
            return helper(node.left, lower, node.val) and helper(node.right, node.val, upper)
        return helper(root)
       
```      
### Solution 2: Iteration
> **Explanation**
>> We can convert the above solution into iteration with the help of stack.

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N). 

```python
class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
            
        stack = [(root, float('-inf'), float('inf'))] 
        while stack:
            root, lower, upper = stack.pop()
            if not root:
                continue
            val = root.val
            if val <= lower or val >= upper:
                return False
            stack.append((root.right, val, upper))
            stack.append((root.left, lower, val))
        return True 
       
```       

### Solution 3: Inorder traversal
> **Explanation**
>> If the result of inorder traversal is not sorted, then this is not a valid BST. And in order to save space, we don't need to store the whole tree. We can just check whether each element in inorder is smaller than the next one or not.

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N), for we still need stack.

```python
class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        stack, inorder = [], float('-inf')
        
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            # If next element in inorder traversal
            # is smaller than the previous one
            # that's not BST.
            if root.val <= inorder:
                return False
            inorder = root.val
            root = root.right

        return True
       
```   

## 105. Construct Binary Tree from Preorder and Inorder Traversal (Median)
- **Link:** https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/


### Solution 1: Recursion
> **Explanation**
>> The first element in the preorder is the root. Then, the elements before the root in the inorder belong to the left tree. And the rest belong to the right tree. Therefore, we can built the tree recursively.
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N). 

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder[0])  
        root.left = self.buildTree(preorder[1:root_index+1], inorder[:root_index])
        root.right = self.buildTree(preorder[root_index+1:], inorder[root_index+1:])
        
        return root       
```   
> The slicing is quite expensive. We can use help function to avoid slicing.

### Solution 2: Iteration
> **Explanation**
>> **Algorithm**
>>> 1. Keep pushing the nodes from the preorder into a stack (and keep making the tree by adding nodes to the left of the previous node) until the top of the stack matches the inorder.
>>> 2. At this point, pop the top of the stack until the top does not equal inorder (keep a flag to note that you have made a pop).
>>> 3. Repeat 1 and 2 until preorder is empty. The key point is that whenever the flag is set, insert a node to the right and reset the flag.(referring to @[gpraveenkumar](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/discuss/34555/The-iterative-solution-is-easier-than-you-think!))

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N).

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        if len(preorder) == 0:
            return None
            
        head = TreeNode(preorder[0])
        stack = [head]
        i = 1
        j = 0
        
        while i < len(preorder):
            temp = None
            t = TreeNode(preorder[i])
            while stack and stack[-1].val == inorder[j]:
                temp = stack.pop()
                j += 1
            if temp:
                temp.right = t
            else:
                stack[-1].left = t
            stack.append(t)
            i += 1
        
        return head
```

## 102. Binary Tree Level Order Traversal (Median)
- **Link:** https://leetcode.com/problems/binary-tree-level-order-traversal/

### Solution: Breadth-First-Search (Queue)
> **Explanation**
>> Simple level traversal.
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N). 

```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return None
        # first-in-first-out
        queue = [root]
        res = []
        
        while len(queue)>0:
            cur_level, size = [], len(queue)
            for i in range(size):
                # pop out the first element in the queue
                node = queue.pop(0)
                cur_level += [node.val]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res += [cur_level]
        return res  
```   

## 94. Binary Tree Inorder Traversal (Median)
- **Link:** https://leetcode.com/problems/binary-tree-inorder-traversal/

### Solution 1: Recursion
> **Complexity Analysis**  
>> - Time complexity : O(N), since T(n)=2⋅T(n/2)+1.
>> - Space complexity : O(N). 

```python
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if root:
            if self.inorderTraversal(root.left):
                res += self.inorderTraversal(root.left)
            res.append(root.val)
            if self.inorderTraversal(root.right):
                res += self.inorderTraversal(root.right)
        return res

```   
### Solution 2: Iteration
> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N). 

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

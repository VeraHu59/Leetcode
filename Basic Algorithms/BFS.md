# Breadth-First Search

BFS can be used in two typical questions: 
1. Traversal in graph 
- Level Order Traversal
- Connected Component
- Topological Sorting
2. Shortest path in simple graph (no direction and the length of edge is 1)

## Algorithm
### BFS in Tree
1. Create a queue and put the first node into the queue
2. While queue is not empty, add the child nodes of current node into the queue
3. Traverse each level
### BFS in Graph
As there is cycle in graph, we need to use hash map to decide whether we have complete exploration of this node or not.

## Sample Question

### 102. Binary Tree Level Order Traversal (Median)
- **Link:** https://leetcode.com/problems/binary-tree-level-order-traversal/

```python
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # corner case
        if not root:
            return None
        # queue: first-in-first-out
        queue = [root]
        res = []
        # while queue is not null
        while len(queue)>0:
            cur_level, size = [], len(queue)   # level x
            for i in range(size):
                # pop out the first element in the queue
                node = queue.pop(0)
                cur_level += [node.val]
                if node.left:
                    queue.append(node.left)   # level x+1
                if node.right:
                    queue.append(node.right)
            res += [cur_level]
        return res  
```   


## Other Typical Questions

### Serialize and Deserialize Binary Tree
- **Link:** https://www.lintcode.com/problem/serialize-and-deserialize-binary-tree/description

Serialization means transfer object into string. Deserialization means transfer string to object.
```python
# BST Serialization and Deserialization
class Codec:
    def serialize(self, root):
        if not root:
            return ''
        queue = [root]
        res = []
        while queue:
        	node = queue.pop(0)
        	if node:
        		# this is not the same with normal BST
        		# we need to push the null into queue 
        		queue.append(node.left)
        		queue.append(node.right)
        	res.append(str(node.val) if node else '#')
        return ','.join(res)

                
    
    def deserialize (self, data):
    	if not data:
    		return None
    	data = data.split(',')
    	root = TreeNode(int(data[0]))
    	queue = [root]
        index = 1
    	while queue:
    		node = queue.pop(0)
    		if data[index] != '#':
    			node.left = TreeNode(int(data[index]))
    			queue.append(node.left)
    		index += 1
    		if data[index+1] != '#':
    			node.right = TreeNode(int(data[index+1]))
    			queue.append(node.right)
    		index += 1
    	return root
```

### Graph Valid Tree
Two conditions: (1) N nodes, N-1 edges (2) Every node is connected with each other (Connected component)



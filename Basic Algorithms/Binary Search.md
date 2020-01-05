# Binary Search

Given a sorted array - nums, and a target - integer. Find the **any/first/last** position of target in nums. Return -1 if target
does not exist.

There are several implicit cases that can apply in Binary Search Problems. For example, OOXX problem (find first X or last O). 
Or there is no OOXX, but we can use some condition to remain the half that contains the target.

Two Common Time Complexity of Binary Search:
1. T(N) = T(N/2) + O(1) = O(logN)
2. T(N) = T(N/2) + O(N) = O(N)

Time Complexity in Code Interview:
- O(logN) - Binary Search
- O(N^1/2) - Decomposing Mass Factor
- O(N) - Frequent!
- O(NlogN) - Usually need to sort
- O(N^2) - Dynamic Programming, Array, Enumeration
- O(N^3) - Dynamic Programming, Array, Enumeration
- O(2^N) - Combination
- O(N!) - Permutation

If you need to optimize an obvious O(N) problem, the method you use is usually Binary Search.

Question: Recursion or While Loop?

When it is possible, you should not use recursion. For recursion is not a good coding pattern, it may lead to stack overflow. You can discuss this with interviewer.


## Sample Code
```python
def Binary_Search(nums, target):
    start, end = 0, len(nums)-1
    while start+1 < end:
        # in case of overflow
        mid = start + (end - start)/2
        if nums[mid] == target:
            end = mid
        elif nums[mid]> target:
            end = mid
        else:
            start = mid
        
        if nums[end] == target:
            return end
        if nums[start] == target:
            return start
    return -1
```

```python
def Binary_Search(nums, target):
    start, end = 0, len(nums)-1
    while start<=end:
        # in case of overflow
        mid = start + (end - start)/2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
            
    return -1
```

# Questions

Here is the list of questions which use the knowledge of Array.

- [1 Two Sum](https://leetcode.com/problems/two-sum/)([Look at Array.md](https://github.com/VeraHu59/Leetcode/blob/master/Questions/Array.md))
- [136 Single Number](https://leetcode.com/problems/single-number/)
- [202 Happy Number](https://leetcode.com/problems/happy-number/)

# Solutions

## 136. Single Number (Easy)

- **Link:** https://leetcode.com/problems/single-number/

### Solution 1: Math
> **Explanation**
>> This idea is quite intuitive.  
>
> **Complexity Analysis**  
>> - Time complexity : O(2N) = O(N). 
>> - Space complexity : O(N).
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return 2*sum(set(nums))-sum(nums)
```

### Solution 2: Bit Manipulation
> **Explanation**
>> As a^a=0, XOR is ofter used when we want to find value that appear odd times.
>
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(1).
```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for i in nums:
            a ^= i
        return a
```

## 202. Happy Number (Easy)

- **Link:** https://leetcode.com/problems/happy-number/

### Solution 1: Hash set
> **Explanation**
>> If the sum of digits is not in the hash set, then we add it into the set. Otherwise, it means the loop is endless(cyclic). In this case, we return False.  
>
> **Complexity Analysis**  
>> - Time complexity : O(logN).
>> - Space complexity : O(logN).
```python
class Solution:
    def isHappy(self, n: int) -> bool:
    	# Introduction of set: https://www.runoob.com/python3/python3-set.html
        member = set()
        while(n!=1):
        	# calculate the sum of squares of each digit
            n = sum([int(i)**2 for i in str(n)])
            # n in member means this is an endless loop, it will repeat the last circle again
            if n in member:
                return False
            else:
                member.add(n)
        
        return True
```

### Solution 2: Two Pointers
> **Explanation**
>> We need to detect if there is cycle in the linked list. And what we do is using the idea of slow and fast pointers(often use to detect cycle). If the fast and slow value are the same, it means that the value converges.
>
> **Complexity Analysis**  
>> - Time complexity : O(logN). 
>> - Space complexity : O(1). In this case, we don't use hash set to detect cycle. Two pointers take constant space.
```python
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    r1 = self.step(n)
    r2 = self.step(r1)
    while(r1 != 1):
        if(r1 == r2):
            return False
        else:
            r1 = self.step(r1)
            r2 = self.step(self.step(r2))
    return True

def step(self, n):
    result = 0
    while(n):
        result += pow(n % 10, 2)
        n = n // 10
    return result
```

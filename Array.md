# Questions

Here is the list of questions which use the knowledge of Array.

- [1 Two Sum](https://leetcode.com/problems/two-sum/)
- [53 Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
- [26 Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [66 Plus One](https://leetcode.com/problems/plus-one/)

# Solutions

## 1. Two Sum (Easy)

- **Link:** https://leetcode.com/problems/two-sum/ 

### Solution 1: Brute Force
> **Explanation**
>> The brute force idea is quite intuitive. We can loop through each element x and find if there is another value that equals to target - x.
>
> **Complexity Analysis**  
>> - Time complexity : O(N^2). 
>> - Space complexity : O(1).
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(0,len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:   
                    h = [i,j]
                    return h
```

### Solution 2: Two-pass Hash Table
> **Explanation**
>> We can reduce the time complexity from O(N^2) to O(N) by trading space for speed.A simple implementation uses two iterations. In the first iteration, we add each element's value and its index to the table. Then, in the second iteration we check if each element's complement (target - nums[i]) exists in the table. Beware that the complement must not be nums[i] itself!
>
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N).


```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # initialize a dictionary
        h = {}
        # enumerate() allows us to loop over something and have an automatic counter.
        for i, num in enumerate(nums):
            n = target - num
            # whether n is in the index of dictionary h
            if n not in h:
                h[num] = i
            else:
                return [h[n], i]
```
> Keep in mind that the time complexity of delete, get item, set item, contains of dictionary are all O(1). And the time complexity of find, insert, delete operations in hash table are all O(1)


## 53. Maximum Subarray (Easy)

- **Link:** https://leetcode.com/problems/maximum-subarray/ 

### Solution 1: Dynamic Programming
> **Explanation**
>> The idea is: if the maxSubarray of first i-1 elements is positive, then we can add that value to nums[i]. Otherwise, we set it to 0.
>> And we don't need all the maxSubarray(nums,i). What we need is just maxSubarray(nums,i-1) so as we can do the comparison.
>> If we meet DP problem, the first problem comes out to our mind should be: what's the statement of the sub-problem, whose format should satisfy that if we've solved a sub-problem, it would be helpful for solving the next-step sub-problem, and, thus, eventually helpful for solving the original problem.
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(1).

```python 
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1,len(nums)):
             nums[i] = max(nums[i], nums[i-1] + nums[i])
        return max(nums)
```

### Solution 2: Divide and Conquer
> **Explanation**
>> The Divide-and-Conquer algorithm breaks nums into two halves and find the maximum subarray sum in them recursively. Well, the most tricky part is to handle the case that the maximum subarray spans the two halves. For this case, we use a linear algorithm: starting from the middle element and move to both ends (left and right ends), record the maximum sum we have seen. In this case, the maximum sum is finally equal to the middle element plus the maximum sum of moving leftwards and the maximum sum of moving rightwards. (from [jianchao-li
](https://leetcode.com/problems/maximum-subarray/discuss/20452/C%2B%2B-DP-and-Divide-and-Conquer))
>
> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N).


```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left, right = 0, len(nums)-1
        return self.helper(nums, left, right)
    
    def helper(self, nums, left, right):
        mid = left+(right-left)//2
        # calculate the max subarray of left side
        l_max = self.helper(nums, left, mid-1)
        # calculate the max subarray of right side
        r_max = self.helper(nums, mid+1, right)
        
        sum_l, sunm_r, ml, mr = 0, 0, 0, 0
        # if the subarray span the two halves
        for i in range(mid-1,-1,-1):
            sum_l += nums[i]
            ml = max(ml,sum_l)
        
        for j in range(mid,right+1):
            sum_r += nums[j]
            mr = max(mr,sum_r)
        
        return max(max(l_max, r_max), mr+ml+nums[mid])        
```

## 26. Remove Duplicates from Sorted Array (Easy)

- **Link:** https://leetcode.com/problems/remove-duplicates-from-sorted-array/

### Solution: Two Pointers
> **Explanation**
>> Since the array is already sorted, we can keep two pointers i and j, where i is the slow-runner while j is the fast-runner. 
>> As long as nums[i] = nums[j], we increment jj to skip the duplicate.
>> If nums[j] <> nums[i], it means that we have dropped all duplicated, so we must copy its value to nums[i + 1]. i is then incremented and we repeat the same process again until j reaches the end of array.(from leetcode solution)

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(1).

```python 
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        # There are two pointers, i works as slow runner, and j works as fast runner
        i = 0
        for j in range(1,len(nums)):
            if nums[j] != nums[i]:
                i+=1
                nums[i] = nums[j] #nums[i] represents the new list without duplicates
            else:
                pass
                
        return i+1
```

## 66. Plus One (Easy)

- **Link:** https://leetcode.com/problems/plus-one/

### Solution 1: Brute Force
> **Explanation**
>> We trasfer the string into integer and add 1 to that integer.

> **Complexity Analysis**  
>> - Time complexity : O(3*N) = O(N). 
>> - Space complexity : O(N).

```python 
def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        nums = 1
        for i in range(n):
            nums += digits[i]*pow(10,n-i-1)
        
        new_digits = []
        while(nums):
            new_digits.append(nums % 10)
            nums = nums // 10
            
        new_digits.reverse()
        
        return new_digits
```
> Note that in-built reverse() has no return value, it makes change on the original list, and the complexity of reverse is O(N).

### Solution 2: Recursion
> **Explanation**
>> If the last digit is not 9, we can simply add 1 to the last digit. Otherwise, we call the recursive function.

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N^2).

```python 
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        # the base case
        if len(digits) == 1 and digits[0] == 9:
            return [1, 0]

        if digits[-1] != 9:
            digits[-1] += 1
            return digits
        else:
            digits[-1] = 0
            digits[:-1] = self.plusOne(digits[:-1])
            return digits 
```

### Solution 3

```python 
class Solution:
    def plusOne(self, digits):
        for i in range(len(digits)):
            # ~i gives i-th element from the back
            if digits[~i] < 9:
                digits[~i] += 1
                return digits
            digits[~i] = 0
        return [1] + [0] * len(digits) 
```

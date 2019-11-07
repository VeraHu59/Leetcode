# Questions

Here is the list of questions which use the knowledge of Array.

- [1 Two Sum](https://leetcode.com/problems/two-sum/)
- [53 Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
- [26 Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [66 Plus One](https://leetcode.com/problems/plus-one/)
- [88 Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
- [118 Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)
- [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
- [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- [169 Majority Element](https://leetcode.com/problems/majority-element/)
- [189 Rotate Array](https://leetcode.com/problems/rotate-array/)
- [217 Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- [268 Missing Number](https://leetcode.com/problems/missing-number/)
- [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/solution/)

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

## 88. Merge Sorted Array (Easy)

- **Link:** https://leetcode.com/problems/merge-sorted-array/

### Solution
> **Explanation**
>> It is quite simple if we use sort() function. But if we don't want to use sort(), we can use the idea of two pointers. Compare the elements nums1[i] and nums2[j] and put the smaller one into array (forward). We can also do this backward, in that case we put the larger one into the array.

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(1).

```python 
def merge(self, nums1, m, nums2, n):
        k = m+n-1
        
        while(n>0):
            if m <= 0 or nums1[m-1] <= nums2[n-1]:
                nums1[k] = nums2[n-1]
                n -= 1
                k -=1
            else:
                nums1[k] = nums1[m-1]
                m -= 1
                k -=1
```

## 118. Pascal's Triangle (Easy)

- **Link:** https://leetcode.com/problems/pascals-triangle/

### Solution 1: Dynamic Programming
> **Explanation**
>>  Any row can be constructed using the offset sum of the previous row.(from [sherlock321](https://leetcode.com/problems/pascals-triangle/discuss/38128/Python-4-lines-short-solution-using-map.))

> **Complexity Analysis**  
>> - Time complexity : O(N), where N is the numRows.
>> - Space complexity : O(N), where N is the numRows.

```python 
def generate(self, numRows):
        res = [[1]]
        for i in range(1, numRows):
            # Any row can be constructed using the offset sum of the previous row.
            # map(function, iterable)
            res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
            # use lambda function, list1+list2 equals to sum(list1,list2), but not extend
        return res[:numRows]
```

### Solution 2:
> **Explanation**
>> Although the algorithm is very simple, the iterative approach to constructing Pascal's triangle can be classified as dynamic programming because we construct each row based on the previous row.

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(N).

```python 
def generate(self, numRows):
        res = [[1]]
        for i in range(1, numRows):
            # Any row can be constructed using the offset sum of the previous row.
            # map(function, iterable)
            res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
            # use lambda function, list1+list2 equals to sum(list1,list2), but not extend
        return res[:numRows]
```

# 122. Best Time to Buy and Sell Stock II (Easy)

- **Link:** https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

### Solution
> **Explanation**
>>  The simple idea is that if the stock price is going to decrease tomorrow, we will sell it today until it is going to increase.

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(1).

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        for i in range(1,len(prices)):
            daily_profit = prices[i]-prices[i-1]
            if daily_profit>0:
                max_profit += daily_profit
        
        return max_profit
```

# 121. Best Time to Buy and Sell Stock (Easy)

- **Link:** https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

### Solution
> **Explanation**
>> We need to find out the maximum difference (which will be the maximum profit) between two numbers in the given array.

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(1).

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # Pay attention to special cases
        if len(prices)<=1:
            return 0
        min_price = prices[0]
        max_profit = 0
        for price in prices:
            min_price = min(min_price,price)
            max_profit = max(max_profit, price - min_price)
        
        return max_profit
```

# 169. Majority Element (Easy)

- **Link:** https://leetcode.com/problems/majority-element/

### Solution 1: Sort

> **Complexity Analysis**  
>> - Time complexity : O(NlogN).
>> - Space complexity : O(1).

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]
```

### Solution 2: Hash

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution:
    def majorityElement(self, nums):
        # A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary value
        # Counter is a subclass of dictionary
        counts = collections.Counter(nums)
        # max(iterable, key): key is optional and comparison is performed based on its return value
        # dict.get: get the value for the specified key if key is in dictionary.
        return max(counts.keys(), key=counts.get)
```

# 217. Contains Duplicate (Easy)

- **Link:** https://leetcode.com/problems/contains-duplicate/

### Solution 1: Set
> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution(object):
    def containsDuplicate(self, nums):
        return len(nums) != len(set(nums)) 
```

> We can see set is useful when we deal with array problems. In python, set is similar with dictionary.


### Solution 2: Hash

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        counts = collections.Counter(nums)
        for count in counts.values():
            if count>1:
                return True
        return False
```

# 268. Missing Number (Easy)

- **Link:** https://leetcode.com/problems/missing-number/

### Solution 1: Hash Set
> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums_set = set(nums)
        n = len(nums) + 1
        for i in range(n):
            # the time complexity of find in set is O(1), the worst case is O(n)
            # the time complexity of find in list is O(n)
            if i not in nums_set:
                return i
```


### Solution 2: Gauss Formula

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution:
    def missingNumber(self, nums):
        # calculate the expected sum
        expected_sum = len(nums)*(len(nums)+1)//2 # O(1)
        actual_sum = sum(nums) # O(n)
        return expected_sum - actual_sum
```


### Solution 3: Bit Manipulation

> **Explanation**
>> For value appears in the list, we can find a match of value-index, where value == index. Then we know that a^a=0, 0^a=a. Therefore, we will be left with the missing number. 

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(1).

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        missing = len(nums)
        for i,num in enumerate(nums):
            missing ^= i^num
        
        return missing
```

# 283. Move Zeroes (Easy)

- **Link:** https://leetcode.com/problems/move-zeroes/

### Solution 
> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(1).

```python
class Solution:
    def moveZeroes(self, nums):
        # we need to start iteration from the last element, otherwise, we will pop the 0 we add.
        for i in range(len(nums))[::-1]:
            if nums[i] == 0:
                nums.pop(i)
                nums.append(0)
```

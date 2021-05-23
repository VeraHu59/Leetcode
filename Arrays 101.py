## 1089. Duplicate Zeros

class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        # count zeros
        dup_num = 0
        length = len(arr)
        for i in range(length+1):
            # we can discard the elements after the boundary
            if i > length - dup_num - 1:
                break
                
            if arr[i] == 0:
                # edge case: if zero is on the boundary, this zero cannot be replicated
                if i == length - dup_num - 1:
                    # set the last element to zero
                    arr[length-1] = 0
                    length -= 1 
                    break
                dup_num += 1
            else:
                pass
        
        # replicate zeros
        last = length - dup_num - 1
       
        res_last = length - 1 
        for j in range(last, -1, -1):
            if arr[j] == 0:
                arr[res_last] = 0 
                res_last -= 1
                arr[res_last] = 0 
                res_last -= 1
            else: 
                arr[res_last] = arr[j]
                res_last -= 1


## 88. Merge Sorted Array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        
        # the index of the largest number
        last = m + n - 1 
        i, j = m-1, n-1
        while i >= 0 and j >= 0:
            if nums1[i] >= nums2[j]:
                nums1[last] = nums1[i]
                i -= 1
                last -= 1
            else:
                nums1[last] = nums2[j]
                j -= 1
                last -= 1 
        # if the iteration of nums1 is complete, but that of nums2 is incomplete
        if j >= 0:
            nums1[:j+1] = nums2[:j+1]
        
        return nums1


## 27. Remove Element

# Solution 1: Two Pointers (when elements to remove are not rare)
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        j = 0
        # i is the fast pointer, j is the slow pointer
        for i in range(len(nums)):
            if nums[i] == val:
                pass
            else:
                nums[j] = nums[i]
                j += 1

        return j

# Solution 2: Swap elements (when the element to remove are rare)
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        length = len(nums)
        
        for i in range(length):
            if nums[i] == val:
            	# if the last element does not equal val, swap them
                while(length-1 >= i):
                    if nums[length-1] != val:
                        nums[i] = nums[length-1]
                        length -= 1
                        break
                    else:
                        length -= 1
            else:
                pass
            
        return length

## 26. Remove duplicates from the sorted array

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 1
        
        for i in range(1,len(nums)):
            if nums[i] != nums[i-1]:
                nums[j] = nums[i]
                j += 1
            else:
                pass
            
        return j

## 1346. Check If N and Its Double Exist

class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        # edge case
        if not arr or len(arr) == 0:
            return False
        
        if arr.count(0) > 1:
            return True
        
        double_arr = [elem*2 for elem in arr]

        for elem in double_arr:
        	if elem != 0 and elem in arr:
        		return True
        return False

## 941. Valid Mountain Array

class Solution:
    def validMountainArray(self, A: List[int]) -> bool:
        n = len(A)
        if n < 3 or not A:
            return False
        
        i = 0
        while i+1 < n and A[i] < A[i+1]:
            i += 1
        
        if i == n-1 or i == 0:
            return False
        
        while i+1 < n and A[i] > A[i+1]:
            i += 1
        
        return i == n-1


## 1299. Replace Elements with Greatest Element on Right Side

class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        last = len(arr) - 1
        r_max = arr[last]
        arr[last] = -1
        
        for i in range(last-1, -1, -1):
            curr = arr[i]
            arr[i] = r_max
            r_max = max(r_max, curr)
            
        return arr

## 283. Move Zeroes

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        n = len(nums)
        i = 0
        
        for j in range(n):
            if nums[j] != 0:
                nums[i] = nums[j]
                i += 1
            else:
                pass
        
        if i != n:
            nums[i:] = [0]*(n-i)
            
        return nums

## 905. Sort Array By Parity

class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        left = 0
        right = len(A) - 1
        
        while left < right:
            if A[left] % 2 == 0 and A[right] % 2 == 1 :
                left += 1
                right -= 1
            elif A[left] % 2 == 0 and A[right] % 2 == 0:
                left += 1
            elif A[left] % 2 == 1  and A[right] % 2 == 1:
                right -= 1
            else:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
        
        return A

## 1051. Height Checker

class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        sort_heights = sorted(heights)
        res = 0
        for i in range(len(heights)):
            if heights[i] != sort_heights[i]:
                res += 1
        return res


## 414. Third Maximum Number

# Solution 1: Hash Set
# Use hash map to delete duplicates O(N)
# delete the maximum, second maximum values O(N+N)
# return the remaining maximum value
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        nums = set(nums)
        
        maximum = max(nums)
        
        if len(nums) < 3:
            return maximum
        
        nums.remove(maximum)
        
        second_max = max(nums)
        
        nums.remove(second_max)
        
        return max(nums)

# Solution 2
# iterate through the array and keep the top 3 elements in a set
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        max_set = set()
        
        for num in nums:
            max_set.add(num)
            if len(max_set)>3:
                max_set.remove(min(max_set))
        if len(max_set)<3:
            return max(max_set)
        return min(max_set)

## 448. Find all Numbers Disappeared in an Array
# Solution 1: Hash map
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        hash_table = {}
        for num in nums:
            hash_table[num] = 1
       
        res = []
        for i in range(1,len(nums)+1):
            if i in hash_table:
                pass
            else:
                res.append(i)
                
        return res

# Solution 2: Make use of index

class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for num in nums:
            new_index = abs(num) - 1
            if nums[new_index] < 0:
                continue
            else:
                nums[new_index] = nums[new_index]*-1
        res = []
        for index,num in enumerate(nums):
            if num > 0:
                res.append(index+1)
        
        return res

# Solution 3: Python set operations
# operations: union(), difference(), intersection()
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        num=set(nums)
        rng=set(range(1,len(nums)+1))
        return list(rng-num)


## 977. Squares of a Sorted Array

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        # edge case
        if not nums:
            return None
        
        square_nums = [elem**2 for elem in nums]
        
        # two pointers
        n = len(nums)
        left = 0
        right = n-1
        res = [0]*n
        
        for i in range(n-1,-1,-1):
            if square_nums[left] <= square_nums[right]:
                res[i] = square_nums[right]
                right -= 1
            else:
                res[i] = square_nums[left]
                left += 1
        
        return res

## 487. Max Consecutive Ones II
# Solution 1: Brute Force
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        longest_sequence = 0
        for left in range(len(nums)):
            num_zeroes = 0
            for right in range(left, len(nums)):   # check every consecutive sequence
                if num_zeroes == 2:
                    break
                if nums[right] == 0:               # count how many 0's
                    num_zeroes += 1
                if num_zeroes <= 1:                 # update answer if it's valid
                    longest_sequence = max(longest_sequence, right - left + 1)
        return longest_sequence

# Solution 2: Sliding window
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_len = 0
        left = 0
        right = 0
        num_zeros = 0
        while right < len(nums):
            if nums[right] == 0:
                num_zeros += 1
            
            while(num_zeros == 2):
                if nums[left] == 0:
                    num_zeros -= 1
                left += 1
            
            max_len = max(max_len, right-left+1)
            right += 1
        
        return max_len

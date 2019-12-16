# Questions

Here is the list of questions which use the knowledge of Array.

- [1 Two Sum](https://leetcode.com/problems/two-sum/)([Look at Array.md](https://github.com/VeraHu59/Leetcode/blob/master/Questions/Array.md))
- [3 Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [36 Valid Sudoku](https://leetcode.com/problems/valid-sudoku/)
- [49 Group Anagrams](https://leetcode.com/problems/group-anagrams/)
- [94 Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)([Look at Tree.md](https://github.com/VeraHu59/Leetcode/blob/master/Questions/Tree.md))
- [136 Single Number](https://leetcode.com/problems/single-number/)
- [138 Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)([Look at Linked list.md](https://github.com/VeraHu59/Leetcode/blob/master/Questions/Linked%20list.md))
- [202 Happy Number](https://leetcode.com/problems/happy-number/)
- [204 Count Primes](https://leetcode.com/problems/count-primes/)
- [217 Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- [242 Valid Anagram](https://leetcode.com/problems/valid-anagram/)
- [347 Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [350 Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/)
- [387 First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/)

# Solutions

## 3. Longest Substring Without Repeating Characters (Median)

- **Link:** https://leetcode.com/problems/longest-substring-without-repeating-characters/

### Solution: Sliding Window (Hash map)
> **Explanation**

>>  If a substring s_{ij} from index i to j - 1 is already checked to have no duplicate characters. We only need to check if s[j] is already in the substring s_{ij}. If s[j] is not in the substring, we slide the index j to the right. If it is not in the HashSet, we slide j further. Doing so until s[j] is already in the HashSet. At this point, we found the maximum size of substrings without duplicate characters start with index i. If we do this for all i, we get our answer.

>> However, in the worst case, each character will be visited twice by i and j. So the time complexity for Hash Set is O(2N). We can optimize this solution by using Hash Map. Instead of using a set to tell if a character exists or not, we could define a mapping of the characters to its index. Then we can skip the characters immediately when we found a repeated character.

>> In conclusion, slicing window is a very powerful method in array/string.It commonly starts with two indexes.

> **Complexity Analysis**  
>> - Time complexity : O(N). 
>> - Space complexity : O(MIN(M,N)), we need extra space k to store hash set. The size k of the Set is upper bounded by the size of the string N and the size of the charset/alphabet M.

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        used ={}
        start, max_len = 0, 0 
        for i,num in enumerate(s):
            if num in used and start<=used[num]:
                start = used[num]+1
            else:
                max_len = max(max_len,i-start+1)
            used[num] = i
        return max_len
```


## 36. Valid Sudoku (Median)

- **Link:** https://leetcode.com/problems/valid-sudoku/

### Solution: Hash map
> **Explanation**

>> The idea behind this solution is quite easy, we should ensure that:
>> - There is no rows with duplicates.
>> - There is no columns with duplicates.
>> - There is no sub-boxes with duplicates.
>> We use hash map to store the element we have already encountered in each row, column and box.

> **Complexity Analysis**  
>> - Time complexity : O(1), since all we do here is just one iteration over the board with 81 cells.
>> - Space complexity : O(1).

```python
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # initialze hash map to store the elements in the board
        rows, columns, boxs = [{} for _ in range(9)], [{} for _ in range(9)], [{} for _ in range(9)]
        for row in range(9):
            for column in range(9):
                curr_num = board[row][column] 
                box = (row//3)*3+column//3
                # if the element is not '.' and never appear in current row, column and box, add it into the hash table
                if curr_num!='.' and curr_num not in rows[row] and curr_num not in columns[column] and curr_num not in boxs[box]:
                    rows[row][curr_num] = 1
                    columns[column][curr_num] = 1
                    boxs[box][curr_num] = 1
                    
                # otherwise, return false  
                elif curr_num in rows[row] or curr_num in columns[column] or curr_num in boxs[box]:
                    return False
   
        return True
```

## 49. Group Anagrams (Median)

- **Link:** https://leetcode.com/problems/group-anagrams/

### Solution 1: Categorized by Sorted String
> **Explanation**

>>  The idea behind this solution is if we sort two anagrams, the result should be the same.

> **Complexity Analysis**  
>> - Time complexity : O(NKlogK), where K is the maximum length of a string in strs and N is the length of strs. 
>> - Space complexity : O(NK), we need extra space k to store hash map. 

```python
class Solution(object):
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()
```

### Solution 2: Categorized by Count
> **Explanation**

>>  The idea behind this solution is: two strings are anagrams if and only if their character counts (respective number of occurrences of each character) are the same.

> **Complexity Analysis**  
>> - Time complexity : O(NK). 
>> - Space complexity : O(NK). 

```python
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        # Initialize a default dictionary
        ans = collections.defaultdict(list)
        for s in strs:
            # As there are only 26 letters, we initialize a list whose length is 26
            count = [0]*26
            for c in s:
                count[ord(c)-ord('a')]+=1
            # if two string are anagrams, then the key of dictionary ans is the same
            ans[tuple(count)].append(s)
        # return the value of dictionary
        return ans.values()
```

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

## 204. Count Primes (Easy)

- **Link:** https://leetcode.com/problems/count-primes/

### Solution 
> **Explanation**
>> If a number is the power of another number, then this number cannot be a prime. 
>
> **Complexity Analysis**  
>> - Time complexity : O(NlogN).
>> - Space complexity : O(N).

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n<3:
            return 0
        prime = [1]*n
        prime[0], prime[1] = 0, 0
        # all multiples of i cannot be prime
        # list[start:end:step]
        for i in range(2, int(n**(1/2)+1)):  # this loop will run sqrt(n)-1 times
            prime[i*i:n:i] = [0]*len(prime[i*i:n:i]) # (sqrt(n)-1)*logn
        
        return sum(prime)
```

## 217. Contains Duplicate (Easy)

- **Link:** https://leetcode.com/problems/contains-duplicate/

### Solution: Hash

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
> Another method using hash set is also very straightforward.

```python
class Solution(object):
    def containsDuplicate(self, nums):
        return len(nums) != len(set(nums)) 
        # set() return a sorted sequence and no duplicate included  
```


## 242. Valid Anagram(Easy)

- **Link:** https://leetcode.com/problems/valid-anagram/

### Solution 1: Hash

> **Complexity Analysis**  
>> - Time complexity : O(N).
>> - Space complexity : O(N).

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        counts = collections.Counter(s)
        
        for str in t:
            if str not in counts:
                return False
            elif str in counts:
                counts[str] -= 1
                
        for key, value in counts.items():
            if value != 0:
                return False
            
        return True
```
>> Another method using hash set is also very straightforward.

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dict1, dict2 = {},{}
        
        for num in s:
            # get(key, default = None), which returns the value of specific key in the dictionary.
            dict1[num] = dict1.get(num,0)+1
        
        for num in t:
            dict2[num] = dict2.get(num,0)+1
        
        # we can just compare these two dicionaries    
        return dict1 == dict2 
```

### Solution 2: Sorted

> **Complexity Analysis**  
>> - Time complexity : O(NlogN).
>> - Space complexity : O(1).

```python
def isAnagram3(self, s, t):
    return sorted(s) == sorted(t)
```
## 347. Top K Frequent Elements (Median)

- **Link:** https://leetcode.com/problems/top-k-frequent-elements/

### Solution 1: Hash

> **Complexity Analysis**  
>> - Time complexity : O(NlogK). The complexity of Counter method is O(N),the time complexity of most_common is O(NlogK) 
>> - Space complexity : O(N). 

```python
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        count = collections.Counter(nums)
        # most_common() is a method in Counter, which reture the key-value pairs which have top k highest value.
        return [k for k,v in count.most_common(k)]
```

> If we don't use most_common() function in Counter, we can use following sorted method:

```python
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        count = collections.Counter(nums)
        # sorted(dict), which returns the sorted key
        arr = sorted(count,key = count.get, reverse = True)
        return arr[:k]
```

### Solution 2: Heap

> **Complexity Analysis**  
>> - Time complexity : O(NlogK). To build a heap and output list takes O(Nlog(k)). Hence the overall complexity of the algorithm is O(N + N log(k)) = =O(Nlog(k)).
>> - Space complexity : O(N).

```python
class Solution:
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """ 
        count = collections.Counter(nums)   
        return heapq.nlargest(k, count.keys(), key=count.get) 
```

## 350. Intersection of Two Arrays II (Easy)

- **Link:** https://leetcode.com/problems/intersection-of-two-arrays-ii/

### Solution 1: Sorted (Two Pointers)

> **Complexity Analysis**  
>> - Time complexity : O(NlogN).
>> - Space complexity : O(N).

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        length1 = len(nums1)
        length2 = len(nums2)

        res = []
        l1, l2 = 0, 0
        while(l1 < length1 and l2 < length2):
            if nums1[l1] == nums2[l2]:
                res.append(nums1[l1])
                l1 += 1
                l2 += 1
            elif nums1[l1] < nums2[l2]:
                l1 += 1
            else:
                l2 
```

### Solution 2: Hash

> **Complexity Analysis**  
>> - Time complexity : O(2N) = O(N).
>> - Space complexity : O(N).

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    	res = []

    	counts = collections.Counter(nums1)

    	for num in nums2:
    		if counts[num]>0:
    			# notice that the type of num is int, so it is not iterable. You should transfer it into list
    			res += [num]
    			counts[num] -= 1

    	return res
```



## 387. First Unique Character in a String (Easy)

- **Link:** https://leetcode.com/problems/first-unique-character-in-a-string/

### Solution 
> **Explanation**
>> If a number is the power of another number, then this number cannot be a prime. 
>
> **Complexity Analysis**  
>> - Time complexity : O(NlogN).
>> - Space complexity : O(N).

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        list = []
        for str in s:
            list += [str]
            
        h = collections.Counter(list)
                
        res = []
        for key, value in h.items():
            if value == 1:
                res+= [key]
        
        if len(res) == 0:
            return -1
        else:
            for i, l in enumerate(list):
                if l in res:
                    return i
```

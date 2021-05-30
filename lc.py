1436. Destination City
You are given the array paths, where paths[i] = [cityAi, cityBi] means there exists a direct path going from cityAi to cityBi. Return the destination city, that is, the city without any path outgoing to another city.

It is guaranteed that the graph of paths forms a line without any loop, therefore, there will be exactly one destination city.

Example 1:

Input: paths = [["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
Output: "Sao Paulo"
Explanation: Starting at "London" city you will reach "Sao Paulo" city which is the destination city. Your trip consist of: "London" -> "New York" -> "Lima" -> "Sao Paulo".
Example 2:

Input: paths = [["B","C"],["D","B"],["C","A"]]
Output: "A"
Explanation: All possible trips are:
"D" -> "B" -> "C" -> "A".
"B" -> "C" -> "A".
"C" -> "A".
"A".
Clearly the destination city is "A".

class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        outgoing = {}
        for path in paths:
            city_a, city_b = path
            outgoing[city_a] = outgoing.get(city_a, 0) + 1
            outgoing[city_b] = outgoing.get(city_b, 0)

        for city in outgoing:
            if outgoing[city] == 0:
                return city

we are adding 1 to city_a which doesnt really matter because we only care about city_b
and then return the city with 0 since it only shows up once

#########################################################################

1365. How Many Numbers Are Smaller Than the Current Number
Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it. That is, for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].

Return the answer in an array.

Example 1:

Input: nums = [8,1,2,2,3]
Output: [4,0,1,1,3]
Explanation:
For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3).

class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        output = []
        for i in range(len(nums)):
            count = 0
            for j in range(len(nums)):
                print('i=', i, 'j=', j)
                print('nums[i]=', nums[i], 'nums[j]=', nums[j])
                if j != i and nums[j] < nums[i]:
                    count += 1
            output.append(count)
        return output

i= 0 j= 0
nums[i]= 8 nums[j]= 8
i= 0 j= 1
nums[i]= 8 nums[j]= 1
i= 0 j= 2
nums[i]= 8 nums[j]= 2
i= 0 j= 3
nums[i]= 8 nums[j]= 2
i= 0 j= 4
nums[i]= 8 nums[j]= 3
i= 1 j= 0
nums[i]= 1 nums[j]= 8
i= 1 j= 1
nums[i]= 1 nums[j]= 1
i= 1 j= 2
nums[i]= 1 nums[j]= 2
i= 1 j= 3
nums[i]= 1 nums[j]= 2
i= 1 j= 4
nums[i]= 1 nums[j]= 3
i= 2 j= 0
nums[i]= 2 nums[j]= 8

##########################################################################################################

704. Binary Search
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

Example 1:

Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
Example 2:

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1

Constraints:

1 <= nums.length <= 104
-9999 <= nums[i], target <= 9999
All the integers in nums are unique.
nums is sorted in an ascending order.

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        for i, val in enumerate(nums):
            if val == target:
                return i
        return -1

 next solution uses binary search and is more efficient
 start = start index
 end = end index
 mid = divide by 2 , use floor //

if target num is greater than mid, use mid as new start, and opposite until
you find your target

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start = 0
        end = len(nums)-1

        while start < end:
            mid = (start + len(nums)) // 2
            num = nums[mid]
            print(start, mid, end)
            if target == num:
                return mid
            elif target > mid:
                start = mid
            elif target < mid:
                end = mid

        return -1

########################################################################

409. Longest Palindrome
Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

Example 1:

Input: s = "abccccdd"
Output: 7
Explanation:
One longest palindrome that can be built is "dccaccd", whose length is 7.
Example 2:

Input: s = "a"
Output: 1
Example 3:

Input: s = "bb"
Output: 2

Constraints:

1 <= s.length <= 2000
s consists of lowercase and/or uppercase English letters only.

class Solution:
    def longestPalindrome(self, s: str) -> int:
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
            print('1st', char, count[char])

        result = 0
        is_odd = False
        for char in count:
            if count[char] % 2 == 0:
                print('2nd', char, count[char])
                result += count[char]
            else:
                #  need this for 'ccc'
                result += (count[char]-1)
                is_odd = True

        if is_odd:
            result += 1

        return result

1st a 1
1st b 1
1st c 1
1st c 2
1st c 3
1st c 4
1st d 1
1st d 2
2nd c 4
2nd d 2

#########################################################################

1561. Maximum Number of Coins You Can Get
Medium

There are 3n piles of coins of varying size, you and your friends will take piles of coins as follows:

In each step, you will choose any 3 piles of coins (not necessarily consecutive).
Of your choice, Alice will pick the pile with the maximum number of coins.
You will pick the next pile with maximum number of coins.
Your friend Bob will pick the last pile.
Repeat until there are no more piles of coins.
Given an array of integers piles where piles[i] is the number of coins in the ith pile.

Return the maximum number of coins which you can have.

Example 1:

Input: piles = [2,4,1,2,7,8]
Output: 9
Explanation: Choose the triplet (2, 7, 8), Alice Pick the pile with 8 coins, you the pile with 7 coins and Bob the last one.
Choose the triplet (1, 2, 4), Alice Pick the pile with 4 coins, you the pile with 2 coins and Bob the last one.
The maximum number of coins which you can have are: 7 + 2 = 9.
On the other hand if we choose this arrangement (1, 2, 8), (2, 4, 7) you only get 2 + 4 = 6 coins which is not optimal.
Example 2:

Input: piles = [2,4,5]
Output: 4
Example 3:

Input: piles = [9,8,7,6,5,1,2,3,4]
Output: 18

Constraints:
3 <= piles.length <= 10^5
piles.length % 3 == 0
1 <= piles[i] <= 10^4


class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        sorted_piles = sorted(piles)
        end_index = len(piles)-1
        our_value = 0

        for i in range(len(piles) // 3):
            our_value += sorted_piles[end_index-(i*2)-1]
            print('i', i)
            print('our val', our_value)
            print('sorted piles', sorted_piles)
            print('end index - ', end_index)
            print('last_sorted_piles', sorted_piles[end_index-(i*2)-1])
        return our_value

[2,4,1,2,7,8]
i 0
our val 7
sorted piles [1, 2, 2, 4, 7, 8]
end index -  5
last_sorted_piles 7
i 1
our val 9
sorted piles [1, 2, 2, 4, 7, 8]
end index -  5
last_sorted_piles 2

##########################################################################

1472. Design Browser History
You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:

BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
void visit(string url) Visits url from the current page. It clears up all the forward history.
string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. Return the current url after moving back in history at most steps.
string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after forwarding in history at most steps.


Example:

Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

Explanation:
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"

Constraints:

1 <= homepage.length <= 20
1 <= url.length <= 20
1 <= steps <= 100
homepage and url consist of  '.' or lower case English letters.
At most 5000 calls will be made to visit, back, and forward.

class BrowserHistory:

    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current_index = 0

    def visit(self, url: str) -> None:
        self.current_index += 1
        self.history = self.history[0:self.current_index]
        self.history.append(url)

    def back(self, steps: int) -> str:
        self.current_index = max(0, self.current_index-steps)
        return self.history[self.current_index]

    def forward(self, steps: int) -> str:
        self.current_index = min(len(self.history)-1, self.current_index+steps)
        return self.history[self.current_index]

#########################################################################################


1. Two Sum
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(len(nums)):
                print('i', i, nums[i])
                print('j', j, nums[j])
                if i != j and nums[i] + nums[j] == target:
                    return [i, j]
i 0 3
j 0 3
i 0 3
j 1 2
i 0 3
j 2 4
i 1 2
j 0 3
i 1 2
j 1 2
i 1 2
j 2 4

This is a better optimized solution:

    def twoSum(self, nums, target):
        d = {}
        for i, n in enumerate(nums):
            m = target - n
            if m in d:
                return [d[m], i]
            else:
                d[n] = i


121. Best Time to Buy and Sell Stock
Easy
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxProfit = 0
        minPurchase = prices[0]
        for i in range(1, len(prices)):		
            print(prices[i], 'i')
            maxProfit = max(maxProfit, prices[i] - minPurchase)
            print(maxProfit, 'maxprofit')
            minPurchase = min(minPurchase, prices[i])
            print(minPurchase, 'minpur')
        return maxProfit


1 i
0 maxprofit
1 minpur
5 i
4 maxprofit
1 minpur
3 i
4 maxprofit
1 minpur
6 i
5 maxprofit
1 minpur
4 i
5 maxprofit
1 minpur
####################################################################################:w

217. Contains Duplicate
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
 
Example 1:

Input: nums = [1,2,3,1]
Output: true
Example 2:

Input: nums = [1,2,3,4]
Output: false
Example 3:

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
 
Constraints:

1 <= nums.length <= 105
-109 <= nums[i] <= 109

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dup = set(nums)
        if len(dup) != len(nums):
            return True
        return False

######################################################################################
Quicksort example

def quicksort(array):
    if array < 2:
        return array

    pivot = array[0]
    less = [i for i in array[1:] if i <= pivot]
    greater = [i for i in array[1:] if i > pivot]

    return quicksort(less) + [pivot] = quicksort(greater)

##########################################################################################

238. Product of Array Except Self
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

Constraints:
2 <= nums.length <= 105
-30 <= nums[i] <= 30

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        #output array (just fill empty with 1s)
        res = [1] * len(nums)
        #left and right pointers
        lo = 0
        hi = len(nums) - 1
        #product for left and right
        leftProduct = 1
        rightProduct = 1
        #keep going until pointers meet
        while lo < len(nums):
            #add running from the l/r products to these spots
            res[lo] *= leftProduct
            res[hi] *= rightProduct
            #multiply products by current in nums
            rightProduct *= nums[hi]
            leftProduct *= nums[lo]
            #inc/dec pointers
            lo += 1
            hi -= 1
        return res         
################################################################################################

53. Maximum Subarray
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Example 2:

Input: nums = [1]
Output: 1
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
 
Constraints:
1 <= nums.length <= 3 * 104
-105 <= nums[i] <= 105

 class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        current_sum = nums[0]
        max_sum = current_sum
        for i in range(1, len(nums)):
            current_sum = max(nums[i] + current_sum, nums[i])
            print('cur..sum', current_sum)
            max_sum = max(current_sum, max_sum)
        return max_sum
               
[-2,1,-3,4,-1,2,1,-5,4]
cur..sum 1
cur..sum -2
cur..sum 4
cur..sum 3
cur..sum 5
cur..sum 6
cur..sum 1
cur..sum 5

###########################################################################################

152. Maximum Product Subarray
Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.
It is guaranteed that the answer wijj fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.
Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
 

Constraints:

1 <= nums.length <= 2 * 104
-10 <= nums[i] <= 10
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

class Solution:
    global_max = prev_max = prev_min = nums[0]
    for num in nums[1:]:
        curr_min = min(prev_max*num, prev_min*num, num)
        curr_max = max(prev_max*num, prev_min*num, num)
        global_max= max(global_max, curr_max)
        prev_max = curr_max
        prev_min = curr_min
    return global_max


################################################################################################
Anytime you see a sorted array, you should think  BINARY SEARCH!!!

153. Find Minimum in Rotated Sorted Array
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.
Example 1:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        
        while left < right:
            midpoint = left + right // 2
            print('mid', nums[midpoint])
            print('right', nums[right])
            print('left', nums[left])
            
            if nums[midpoint] > 0 and nums[midpoint] < nums[midpoint -1]:
                return nums[midpoint]
            elif nums[left] <= nums[midpoint] and nums[midpoint] > nums[right]:
                left = midpoint - 1
            else:
                right = midpoint + 1
        return nums[left]
 [3,4,5,1,2]
mid 5
right 2
left 3
mid 1
right 2
left 4           
        
##################################################################################################################################################################

33. Search in Rotated Sorted Array
Medium

!!!     Another Binary Search, we want to find the pivot point (or the smallest element, which would be the pivot), then figure out what side to use binary search on.

There is an integer array nums sorted in ascending order (with distinct values).
Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1

Constraints:
1 <= nums.length <= 5000
-104 <= nums[i] <= 104
All values of nums are unique.
nums is guaranteed to be rotated at some pivot.
-104 <= target <= 104

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        
        while left < right:
            midpoint = (left + right) // 2
            
            if nums[midpoint] > nums[right]:
                left = midpoint + 1
            else:
                right = midpoint
        
        start = left
        left = 0    
        right = len(nums) - 1
        
        if target >= nums[start] and target <= nums[right]:
            left = start
        else:
            right = start
        
        while left <= right:
            midpoint = (left + right) // 2
            
            if nums[midpoint] == target:
                return midpoint
            elif nums[midpoint] > target:
                right = midpoint - 1
            else:
                left = midpoint + 1
                
        return -1

################################################################################################################

15. 3Sum
Medium
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []
Example 3:

Input: nums = [0]
Output: []
 
Constraints:
0 <= nums.length <= 3000
-105 <= nums[i] <= 105

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        
        triplets = []
        
        for i in range(len(nums) - 2):
            
            # this is for non distinct 
            if i > 0 and nums[i] == nums[i -1]: continue

            left = i + 1
            right = len(nums) - 1
            
            while left < right:
                
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum == 0:
                    triplets.append([nums[i], nums[left], nums[right]])
                    
                    
                    # next two are for non distinct too
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                
                elif current_sum < 0:
                    left += 1
                
                else:
                    right -= 1
        return triplets

##################################################################################################################
11. Container With Most Water   #   TWo pointer prob
Medium
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.
Example 1:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1
Example 3:

Input: height = [4,3,2,1,4]
Output: 16
Example 4:

Input: height = [1,2,1]
Output: 2
 
Constraints:
n == height.length
2 <= n <= 105
0 <= height[i] <= 104

class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0
        left = 0
        right = len(height) - 1
        
        while left < right:
            if height[left] < height[right]:
                max_area = max(max_area, height[left] * (right - left))
                left += 1
            else:
                max_area = max(max_area, height[right] * (right - left))
                right -= 1
        return max_area

###########################################################################################################################

371. Sum of Two Integers
Medium
Given two integers a and b, return the sum of the two integers without using the operators + and -.

Example 1:

Input: a = 1, b = 2
Output: 3
Example 2:

Input: a = 2, b = 3
Output: 5
 
Constraints:
-1000 <= a, b <= 1000


class Solution:
    def getSum(self, a: int, b: int) -> int:
        
        # 32 bit mask in hexadecimal
        mask = 0xffffffff
        
        # works both as while loop and single value check 
        while (b & mask) > 0:
            
            carry = ( a & b ) << 1
            a = (a ^ b) 
            b = carry
        
        # handles overflow
        return (a & mask) if b > 0 else a

######################################################################################################3

70. Climbing Stairs   Fibonacci is answer --- 
Easy
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
 
Constraints:

1 <= n <= 45

class Solution:
    def climbStairs(self, n: int) -> int:
        a, b = 1, 1
        for i in range(n):
            a, b = b, a + b
        return a
        
#########################################################################################################

3. Longest Substring Without Repeating Characters     ###  Sliding window example 2
Medium
Given a string s, find the length of the longest substring without repeating characters.

 
Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
Example 4:

Input: s = ""
Output: 0

Constraints:
0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces.


"abcdtabr"

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic, res, start, = {}, 0, 0
        for i, ch in enumerate(s):
            print('i', i, 'ch', ch)
            if ch in dic:
                # check length from start of string to index
                res = max(res, i-start)
                print('res', res)
                # update start of string index to the next index
                start = max(start, dic[ch]+1)
                print('start', start)
            # add/update char to/of dictionary 
            dic[ch] = i
        # answer is either in the begining/middle OR some mid to the end of string
        return max(res, len(s)-start)
i 0 ch a
i 1 ch b
i 2 ch c
i 3 ch d
i 4 ch t
i 5 ch a
res 5
start 1
i 6 ch b
res 5
start 2
i 7 ch r



#  Now using the sliding window
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len, i, j, curr_set, n = 0, 0, 0, set(), len(s)

        while i < n and j < n:
            if s[j] in curr_set:
                curr_set.remove(s[j])
                i += 1
            else:
                curr_set.add(s[j])
                j += 1
            max_len = max(max_len, j - i)
        return max_len


############################################################################################################333

424. Longest Repeating Character Replacement
Medium
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.
Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
 
Constraints:
1 <= s.length <= 105
s consists of only uppercase English letters.
0 <= k <= s.length

condition: keep moving j while the num of replacement is < k (advance j while condition of k is still valid)
    when num of replacment is > k, shrink window my moving i so that we can hav a valid condition again

0 1 2 3
A A B A
j - i + 1 = 4
max_count = 3
j - i + 1 - max_count = the num of chars in the window that are not the char that occurred the most = num of replacements needed

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        i, j, max_count, max_len = 0, 0, 0, 0
        window = [0 for _ in range(26)]
        
        while j < len(s):
            index = ord(s[j]) - ord('A')
            window[index] += 1
            max_count = max(max_count, window[index])
            
            while j - i + 1 - max_count > k:
                window[ord(s[i]) - ord('A')] -= 1
                i += 1
            max_len = max(max_len, j - i + 1)
            j += 1
            
        return max_len 

####################################################################################################################

643. Maximum Average Subarray I   ##   Sliding Window    ## 
Easy
You are given an integer array nums consisting of n elements, and an integer k.
Find a contiguous subarray whose length is equal to k that has the maximum average value and return this value. Any answer with a calculation error less than 10-5 will be accepted.

Example 1:

Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
Example 2:

Input: nums = [5], k = 1
Output: 5.00000o

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        [1,12,-5,-6,50,3]
         ^
         i 
           ^
           j      
        
        '''
        maxSum = windowSum = sum(nums[:k])
        for i in range(k, len(nums)):
            windowSum += nums[i] - nums[i - k]
            maxSum = max(maxSum, windowSum)
        return maxSum / k
        
            
#################################################################################################################
# 
1800. Maximum Ascending Subarray Sum
Easy
Given an array of positive integers nums, return the maximum possible sum of an ascending subarray in nums.

A subarray is defined as a contiguous sequence of numbers in an array.
A subarray [numsl, numsl+1, ..., numsr-1, numsr] is ascending if for all i where l <= i < r, numsi < numsi+1. Note that a subarray of size 1 is ascending.
 
Example 1:

Input: nums = [10,20,30,5,10,50]
Output: 65
Explanation: [5,10,50] is the ascending subarray with the maximum sum of 65.
Example 2:

Input: nums = [10,20,30,40,50]
Output: 150
Explanation: [10,20,30,40,50] is the ascending subarray with the maximum sum of 150.
Example 3:

Input: nums = [12,17,15,13,10,11,12]
Output: 33
Explanation: [10,11,12] is the ascending subarray with the maximum sum of 33.
Example 4:

Input: nums = [100,10,1]
Output: 100
 
Constraints:
1 <= nums.length <= 100
1 <= nums[i] <= 100
        
class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        windowSum = maxSum = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                windowSum += nums[i] 
                maxSum = max(maxSum, windowSum)
            else:
                windowSum = nums[i]
        return maxSum

##############################################################################################################################

1588. Sum of All Odd Length Subarrays
Easy
Given an array of positive integers arr, calculate the sum of all possible odd-length subarrays.
A subarray is a contiguous subsequence of the array.
Return the sum of all odd-length subarrays of arr.
Example 1:

Input: arr = [1,4,2,5,3]
Output: 58
Explanation: The odd-length subarrays of arr and their sums are:
[1] = 1
[4] = 4
[2] = 2
[5] = 5
[3] = 3
[1,4,2] = 7
[4,2,5] = 11
[2,5,3] = 10
[1,4,2,5,3] = 15
If we add all these together we get 1 + 4 + 2 + 5 + 3 + 7 + 11 + 10 + 15 = 58
Example 2:

Input: arr = [1,2]
Output: 3
Explanation: There are only 2 subarrays of odd length, [1] and [2]. Their sum is 3.
Example 3:

Input: arr = [10,11,12]
Output: 66
 
Constraints:
1 <= arr.length <= 100
1 <= arr[i] <= 1000

#####################################################################################################################

242. Valid Anagram
Easy
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
 
Constraints:
1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.
 
Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)
              
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dict1 = {}
        dict2 = {}
        for item in s:
            dict1[item] = dict1.get(item, 0) + 1
        for item in t:
            dict2[item] = dict2.get(item, 0) + 1
        if dict1 == dict2:
            return True
        else:
            return False

##################################################################################################################

76. Minimum Window Substring
Hard
Given two strings s and t of lengths m and n respectively, return the minimum window in s which will contain all the characters in t. If there is no such window in s that covers all characters in t, return the empty string "".
Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Example 2:

Input: s = "a", t = "a"
Output: "a"
 
Constraints:

m == s.length
n == t.length
1 <= m, n <= 105
s and t consist of English letters.

def minWindow(self, s, t):
    need, missing = collections.Counter(t), len(t)
    i = I = J = 0
    for j, c in enumerate(s, 1):
        missing -= need[c] > 0
        need[c] -= 1
        if not missing:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            if not J or j - i <= J - I:
                I, J = i, j
    return s[I:J]

    ###########################################################################################3

49. Group Anagrams
Medium
Given an array of strings strs, group the anagrams together. You can return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Example 2:

Input: strs = [""]
Output: [[""]]
Example 3:

Input: strs = ["a"]
Output: [["a"]]

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = collections.defaultdict(list)
        for string in strs:
            print(groups)
            groups["".join(sorted(string))].append(string)
        return groups.values()

using a dict of all letters in alphabet, groups needs to be a tuple for a key, cant assign list to a key
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = collections.defaultdict(list)
        for string in strs:
            count = [0] * 26
            for c in string:
                count[ord(c) - ord('a')] += 1
            groups[tuple(count)].append(string)
        return groups.values()

##############################################################################################################

20. Valid Parentheses
Easy
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.

Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false
Example 4:

Input: s = "([)]"
Output: false
Example 5:

Input: s = "{[]}"
Output: true

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        lookup = {'}':'{', ')':'(', ']':'['}
        
        for item in s:
            if item in lookup.values():
                stack.append(item)
            elif stack and lookup[item] == stack[-1]:
                stack.pop()
            else:
                return False
        
        return stack == []
               

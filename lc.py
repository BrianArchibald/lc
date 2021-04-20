# 1436. Destination City
# You are given the array paths, where paths[i] = [cityAi, cityBi] means there exists a direct path going from cityAi to cityBi. Return the destination city, that is, the city without any path outgoing to another city.

# It is guaranteed that the graph of paths forms a line without any loop, therefore, there will be exactly one destination city.

# Example 1:

# Input: paths = [["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
# Output: "Sao Paulo"
# Explanation: Starting at "London" city you will reach "Sao Paulo" city which is the destination city. Your trip consist of: "London" -> "New York" -> "Lima" -> "Sao Paulo".
# Example 2:

# Input: paths = [["B","C"],["D","B"],["C","A"]]
# Output: "A"
# Explanation: All possible trips are:
# "D" -> "B" -> "C" -> "A".
# "B" -> "C" -> "A".
# "C" -> "A".
# "A".
# Clearly the destination city is "A".

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

# we are adding 1 to city_a which doesnt really matter because we only care about city_b
# and then return the city with 0 since it only shows up once

#########################################################################

# 1365. How Many Numbers Are Smaller Than the Current Number
# Given the array nums, for each nums[i] find out how many numbers in the array are smaller than it. That is, for each nums[i] you have to count the number of valid j's such that j != i and nums[j] < nums[i].

# Return the answer in an array.

# Example 1:

# Input: nums = [8,1,2,2,3]
# Output: [4,0,1,1,3]
# Explanation:
# For nums[0]=8 there exist four smaller numbers than it (1, 2, 2 and 3).

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

# i= 0 j= 0
# nums[i]= 8 nums[j]= 8
# i= 0 j= 1
# nums[i]= 8 nums[j]= 1
# i= 0 j= 2
# nums[i]= 8 nums[j]= 2
# i= 0 j= 3
# nums[i]= 8 nums[j]= 2
# i= 0 j= 4
# nums[i]= 8 nums[j]= 3
# i= 1 j= 0
# nums[i]= 1 nums[j]= 8
# i= 1 j= 1
# nums[i]= 1 nums[j]= 1
# i= 1 j= 2
# nums[i]= 1 nums[j]= 2
# i= 1 j= 3
# nums[i]= 1 nums[j]= 2
# i= 1 j= 4
# nums[i]= 1 nums[j]= 3
# i= 2 j= 0
# nums[i]= 2 nums[j]= 8

##########################################################################################################

# 704. Binary Search
# Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

# Example 1:

# Input: nums = [-1,0,3,5,9,12], target = 9
# Output: 4
# Explanation: 9 exists in nums and its index is 4
# Example 2:

# Input: nums = [-1,0,3,5,9,12], target = 2
# Output: -1
# Explanation: 2 does not exist in nums so return -1

# Constraints:

# 1 <= nums.length <= 104
# -9999 <= nums[i], target <= 9999
# All the integers in nums are unique.
# nums is sorted in an ascending order.

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        for i, val in enumerate(nums):
            if val == target:
                return i
        return -1

#  next solution uses binary search and is more efficient
#  start = start index
#  end = end index
#  mid = divide by 2 , use floor //

# if target num is greater than mid, use mid as new start, and opposite until
# you find your target

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

# 409. Longest Palindrome
# Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

# Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

# Example 1:

# Input: s = "abccccdd"
# Output: 7
# Explanation:
# One longest palindrome that can be built is "dccaccd", whose length is 7.
# Example 2:

# Input: s = "a"
# Output: 1
# Example 3:

# Input: s = "bb"
# Output: 2
 
# Constraints:

# 1 <= s.length <= 2000
# s consists of lowercase and/or uppercase English letters only.

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
                is_odd = True
        
        if is_odd:
            result += 1
         
        return result

# 1st a 1
# 1st b 1
# 1st c 1
# 1st c 2
# 1st c 3
# 1st c 4
# 1st d 1
# 1st d 2
# 2nd c 4
# 2nd d 2

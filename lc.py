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
                #  need this for 'ccc'
                result += (count[char]-1)
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

#########################################################################

# 1561. Maximum Number of Coins You Can Get
# Medium

# There are 3n piles of coins of varying size, you and your friends will take piles of coins as follows:

# In each step, you will choose any 3 piles of coins (not necessarily consecutive).
# Of your choice, Alice will pick the pile with the maximum number of coins.
# You will pick the next pile with maximum number of coins.
# Your friend Bob will pick the last pile.
# Repeat until there are no more piles of coins.
# Given an array of integers piles where piles[i] is the number of coins in the ith pile.

# Return the maximum number of coins which you can have.

# Example 1:

# Input: piles = [2,4,1,2,7,8]
# Output: 9
# Explanation: Choose the triplet (2, 7, 8), Alice Pick the pile with 8 coins, you the pile with 7 coins and Bob the last one.
# Choose the triplet (1, 2, 4), Alice Pick the pile with 4 coins, you the pile with 2 coins and Bob the last one.
# The maximum number of coins which you can have are: 7 + 2 = 9.
# On the other hand if we choose this arrangement (1, 2, 8), (2, 4, 7) you only get 2 + 4 = 6 coins which is not optimal.
# Example 2:

# Input: piles = [2,4,5]
# Output: 4
# Example 3:

# Input: piles = [9,8,7,6,5,1,2,3,4]
# Output: 18

# Constraints:
# 3 <= piles.length <= 10^5
# piles.length % 3 == 0
# 1 <= piles[i] <= 10^4


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

# [2,4,1,2,7,8]
# i 0
# our val 7
# sorted piles [1, 2, 2, 4, 7, 8]
# end index -  5
# last_sorted_piles 7
# i 1
# our val 9
# sorted piles [1, 2, 2, 4, 7, 8]
# end index -  5
# last_sorted_piles 2

##########################################################################

# 1472. Design Browser History
# You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.

# Implement the BrowserHistory class:

# BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
# void visit(string url) Visits url from the current page. It clears up all the forward history.
# string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. Return the current url after moving back in history at most steps.
# string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after forwarding in history at most steps.


# Example:

# Input:
# ["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
# [["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
# Output:
# [null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

# Explanation:
# BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
# browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
# browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
# browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
# browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
# browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
# browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
# browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
# browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
# browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
# browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"

# Constraints:

# 1 <= homepage.length <= 20
# 1 <= url.length <= 20
# 1 <= steps <= 100
# homepage and url consist of  '.' or lower case English letters.
# At most 5000 calls will be made to visit, back, and forward.

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


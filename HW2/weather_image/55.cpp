#include<bits/stdc++.h>
using namespace std;

// DP 好難 耖
class Solution {
public:
    bool canJump(vector<int>& nums) {
        // DP[i] = x, 從第i格開始跳，最遠可以跳到x
        vector<int> dp;
        for (int i=0; i<nums.size(); i++) {
            dp[i] = i + nums[i];
        }
        for (int i=0; i<nums.size(); i++) {
            for (int j=i+1; i<=i+nums[i]; j++) {
                dp[j] = max(dp[j], dp[i]);
            }
        }
        
    }
};

//greedy
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int max_reach = 0;
        for (int i=0; i<nums.size(); i++) {
            if (i > max_reach) return false;
            max_reach = max(max_reach, i+nums[i]);
            if (max_reach >= nums.size()-1) return true;
        }
        return false;
    }
};

// 2 5 0 0 1
// 2, 3, 1, 1, 4
// 1 0 0 5 1
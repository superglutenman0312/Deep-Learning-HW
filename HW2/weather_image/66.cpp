#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int carry = 1, last = digits.size()-1;
        bool all_nine = true;
        for (int i=0; i<=last; i++) {
            if (digits[i] != 9) all_nine = false;
        }
        if (all_nine) {
            vector<int> ans;
            ans.push_back(1);
            for (int i=0; i<=last; i++) {
                ans.push_back(0);
            }
            return ans;
        }
        for(int i=last; i>=0; i--) {
            digits[i] += carry;
            if (digits[i] == 10) {
                digits[i] = 0;
                carry = 1;
            }
            else {
                carry = 0;
                return digits;
            }
        }
        return digits;
    }
};

class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int carry = 1, last = digits.size()-1;
        for(int i=last; i>=0; i--) {
            digits[i] += carry;
            if (digits[i] == 10) {
                digits[i] = 0;
                carry = 1;
            }
            else {
                carry = 0;
                return digits;
            }
        }
        if (carry) digits.insert(digits.begin(), 1);
        return digits;
    }
};
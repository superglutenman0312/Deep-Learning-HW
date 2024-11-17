#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool isSubsequence(string s, string t) {
        int cnt = 0;
        if (s.empty()) return true;
        for (int i=0; i<t.length(); i++) {
            if (cnt == s.length()) return true;
            if (t[i] == s[cnt]) {
                cnt++;
            }
        }
        return false;
    }
};
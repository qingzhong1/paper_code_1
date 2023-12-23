'''path_test='D:\\cmudata\\CMU-MultimodalSDK-master\\examples\\mmdatasdk_examples\\full_examples\\cmumosi\\test.pkl'
import pickle
data=pickle.load(open(path_test,'rb'))
text=data[1][0][3]
for i in range(len(data)):
    text = data[i][0][3]
    print(' '.join(text))'''
class Solution(object):
    def toeSum(self,nums,target):
        for i in range(len(nums)):
            a=target-nums[i]
            if a in nums:
                y=nums.index(a)
                if y!=i:
                    return [i,y]
                    break
                else:
                    continue
            else:
                continue
class Solution:
    def lengthofLongestSubstring(self, s):
        maxlen = 0
        memo = dict()
        begin, end = 0, 0
        n = len(s)
        while end < n:
            last = memo.get(s[end])
            memo[s[end]] = end
            if last is not None:
                maxlen = max(maxlen, end-begin)
                begin = max(begin, last + 1)
            end += 1
        maxlen = max(maxlen, end-begin)
        return maxlen
''' def LengthOfLongestSubstring(s):
    max=0
    length=len(s)
    if length==1:
        max=1
    else:
        for i in range(length):
            k = 0
            while (i + k <=length):
                text = s[i:i + k]
                k = k + 1
                if len(set(list(text))) == len(list(text)):
                    text_length = len(text)
                    if max < text_length:
                        max = text_length
    return max
str='aus'
max=LengthOfLongestSubstring(str)
print(max)'''
class Solution:
    def lengthofLongestSubstring(self, s):
        maxlen = 0
        memo = dict()
        begin, end = 0, 0
        n = len(s)
        while end < n:
            last = memo.get(s[end])
            memo[s[end]] = end

            if last is not None:
                maxlen = max(maxlen, end-begin)
                begin = max(begin, last + 1)
            end += 1
        maxlen = max(maxlen, end-begin)
        return maxlen

'''a=Solution()
max=a.lengthofLongestSubstring('abcabcabc')
print(max)'''


def LengthOfLongestSubstring(s):
    maxlen=0
    dict_new={}
    begin=0
    end=0
    n=len(s)
    while end<n:
        last=dict_new.get(s[end])
        dict_new[s[end]]=end
        if last is not  None:
            maxlen=max(maxlen,end-begin-1)
            begin=max(begin,last+1)
        end+=1
    maxlen=max(maxlen,end-begin-1)
    return maxlen

def findMedianSortedArrays(nums1,nums2):
    num=nums1+nums2
    num=list(sorted(num))
    length=len(num)
    if length%2==1:
        return num[length//2]
    else:
        return float((num[length//2]+num[length//2-1])/2)
'''def LongestPalindrome(s):
    if len(s)<2:
        return s
    else:
        max_length = 0
        n_length = len(s)
        Pali = []
        Pali_leng = []
        for i in range(n_length):
            for j in range(i, n_length + 1):
                text = s[i:j]
                if list(text) == list(reversed(text)):
                    if max_length < len(text):
                        Pali.append(text)
                        Pali_leng.append(len(text))
        max_s = max(Pali_leng)
        long = Pali[Pali_leng.index(max_s)]
        return long

a=LongestPalindrome('bb')
print(a)'''

def LongestPalindrome(s):
    s_leng=len(s)
    if s_leng<2:
        return s
    else:
        begin=0
        end=0
        max_text=0
        for i in range(s_leng+1):
            for j in range(i):
                if s[i-1]==s[j] and list(reversed(s[j:i]))==list(s[j:i]):
                    if max_text<i-j:
                        max_text=i-j
                        end=i
                        begin=j
        if begin-end==0:
            return s[0]
        else:
            return s[begin:end]
#a=LongestPalindrome('bababa')
#print(a)
#z
'''import torch
a=torch.randn(size=(20,10,768))
b=torch.randn(size=(20,30,30))
import multihead_attention
from multihead_attention import MultiheadAttention
MultiheadAttention(embed_dim=768,num_heads=12,attn_dropout=0.1)
model=multihead_attention.MultiheadAttention(embed_dim=768,num_heads=12)'''

#print(model(a,b,b).shape)
import numpy
dev_predict_list=[-1,-2,-3,-1.1,0,1,0,3]
exclude_zero=True
dev_non_zeros = numpy.array([i for i, e in enumerate(dev_predict_list) if e != 0 or (not exclude_zero)])
print(dev_non_zeros)
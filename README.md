# 余光中诗作深度学习生成器/deep learning model for works of Guangzhong Yu

## Guangzhong Yu is a famous poeist both in mainland China and Taiwan, even through all Chinese communities in the world.

## This model is a deep learning model for automaticaly generating textual sequences after learning 60 pieces of his works. 
The whole process consists of three parts:
1. scrawling selections from the Internet
2. training an LSTM model
3. initialize the input, build the inference model and get the results.

## general params includes:
* batch size : 64
* sequence length : 3
* number of hidden layers : 16
* number of embedding : 16
* number of lstm layers : 1
* num of iterations : 1000
* learning rate : 0.6
* weight dacay : 0.00001

finally you probably get a model with Perp scoring 5 and negative log-likelyhood scoring 1.7.

```
Note that this is model is of token-level, which means the dict is created by Chinese characters cutting. 
The following tasks, e.g. wording embedding and model training, are based on this as well. You can also 
modify the internal functions for a character-level job, and customize all procedures for the 
character-level treatment. But I do recommand the token-level idea, which will bring the faster convergency 
speed and more acceptable results.
```

This model is just for deep learning experiance and memorizing this great man. The params which are suggested by expertise and 
intuition, are absolutly not optimal. Consequently the final output is not guarantteed with Mr. Yu's qualtiy.  

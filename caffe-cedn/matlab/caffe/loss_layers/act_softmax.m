function [act] = act_softmax(pred)

[h,w,c,n] = size(pred);

maxpred = max(pred,[],3);
pred = pred - repmat(maxpred,[1,1,c,1]);

act = exp(pred); 
act = act ./ repmat(sum(act,3),[1,1,c,1]);
function [loss, delta] = loss_crossentropy_paired_softmax_grad(pred, output, penalty)

[h,w,c,n] = size(pred);
maxpred = max(pred,[],3);
pred = pred - repmat(maxpred,[1,1,c,1]);

act = exp(pred); 
act = act ./ repmat(sum(act,3),[1,1,c,1]);

if nargin > 2,
    loss = -(log(act).*output).*penalty;
    loss = sum(loss(:));
    delta = (act - output).*penalty;
else
    loss = -log(act).*output;
    loss = sum(loss(:));
    delta = act - output;
end

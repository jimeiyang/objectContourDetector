function [loss, delta] = loss_crossentropy_paired_sigmoid_grad(pred, output, penalty)

[h,w,c,n] = size(pred);

act = 1 ./ (1 + exp(-pred)); 

if nargin > 2,
    loss = -log(act).*output.*penalty;
    loss = sum(loss(:));
    delta = (act - output).*penalty;
else
    loss = -log(act).*output;
    loss = sum(loss(:));
    delta = act - output;
end

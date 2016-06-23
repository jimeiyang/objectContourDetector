function [loss,grad] = loss_euclidean_grad(pred, output)
% grad = single(mean(pred - output, 4));
% grad = repmat(grad, [1,1,1,num_input]);
% grad = permute(grad, [1,2,4,3]);
loss = sum((pred(:) - output(:)).^2)/2;
grad = single(pred-output);
% grad = grad/size(pred,4);

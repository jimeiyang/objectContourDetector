function [loglikelihood, grad_mu, grad_log_var] = loglikelihood_gaussian_diagonal_covariance(mu, log_var, samples)

[h,w,c,n] = size(mu);

loglikelihood = zeros(n,1,'single');
grad_mu = zeros(size(mu),'single');
grad_log_var = zeros(size(log_var),'single');

for i = 1:n,

  diff = samples(:,:,:,i) - mu(:,:,:,i);
  sigma = exp(.5*log_var(:,:,:,i));
  mahalanobis = bsxfun(@rdivide, diff, sigma).^2;
  mahalanobis = sum(mahalanobis(:));
  loglikelihood(i) = -.5*(h*w*c)*log(2*pi) -sum(log(sigma(:))) -.5*mahalanobis;
  grad_mu(:,:,:,i) = bsxfun(@rdivide, diff, sigma.^2);
  grad_log_var(:,:,:,i) = -.5 + .5*bsxfun(@rdivide, diff, sigma).^2;

end

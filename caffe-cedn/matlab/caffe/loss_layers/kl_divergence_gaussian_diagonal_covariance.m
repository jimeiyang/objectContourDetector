function [kld, grad_mu, grad_log_var] = kl_divergence_gaussian_diagonal_covariance(mu, log_var)
% compute the KL divergence between one gaussian (mu, exp(0.5*log_var)) and another gaussian (0, I)

[h,w,c,n] = size(mu);

kld = zeros(n,1,'single');
grad_mu = zeros(size(mu),'single');
grad_log_var = zeros(size(log_var),'single');

for i = 1 : n
    m = mu(:,:,:,i);
    s = exp(0.5*log_var(:,:,:,i));
    kld(i) = -0.5*sum(sum(sum(1 + log(s.^2) - m.^2 - s.^2)));
    grad_mu(:,:,:,i) = m;
    grad_log_var(:,:,:,i) = 0.5*(-1 + s.^2);
end

function samples = sample_from_gaussian_diagonal_covariance(mu, sigma)

noise = randn(size(sigma));

if numel(sigma) == 1.
    samples = mu + sigma * noise;
else
    samples = mu + sigma .* noise;
end

function [images,masks] = sample_image(im, mask)
% ------------------------------------------------------------------------
IMAGE_DIM = 256;
CROPPED_DIM = 224;

% resize to fixed input size
im = single(im);

if min(size(im,1),size(im,2)) < IMAGE_DIM,
if size(im, 1) < size(im, 2)
    im = imresize(im, [IMAGE_DIM NaN]);
    mask = imresize(mask, [IMAGE_DIM NaN], 'nearest');
else
    im = imresize(im, [NaN IMAGE_DIM]);
    mask = imresize(mask, [NaN IMAGE_DIM], 'nearest');
end
end

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 8, 'single');
masks = zeros(CROPPED_DIM, CROPPED_DIM, 1, 8, 'single');

indices_y = [0:size(im,1)-CROPPED_DIM] + 1;
indices_y = randSample(indices_y, 2);
indices_x = [0:size(im,2)-CROPPED_DIM] + 1;
indices_x = randSample(indices_x, 2);

curr = 1;
for i = indices_y
  for j = indices_x
    images(:, :, :, curr) = im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
    images(:, :, :, curr+4) = images(end:-1:1, :, :, curr);
    masks(:, :, :, curr) = mask(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
    masks(:, :, :, curr+4) = masks(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end

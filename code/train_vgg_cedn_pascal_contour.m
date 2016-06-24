% train semantic segmentation with vggnet fcn
addpath(genpath('../caffe-cedn/matlab'));
model_specs = sprintf('vgg-16-encoder-decoder-%s', 'contour');
use_gpu = true;
model_file = sprintf('%s.prototxt', model_specs);
solver_file = sprintf('%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'lr_policy', 'fixed', 'weight_decay', 0.001, 'solver_type', 3, 'snapshot_prefix', sprintf('../models/PASCAL/%s',model_specs));
make_solver_file(solver_file, model_file, param);
mean_pix = [103.939, 116.779, 123.68];
matcaffe_fcn_vgg_init(use_gpu, solver_file, 0);
weights0 = caffe('get_weights');
vggnet = load('../models/VGG_ILSVRC_16_layers_fcn_model.mat');
for i=1:14, weights0(i).weights = vggnet.model(i).weights; end
caffe('set_weights', weights0);
caffe('set_phase_train');

imnames = textread('../data/PASCAL/train.txt', '%s');
length(imnames)

H = 224; W = 224;

fid = fopen(sprintf('../results/PASCAL/%s-w10-train-errors.txt', model_specs),'w');
for iter = 1 : 30
  tic
  loss_train = 0;
  error_train_mask = 0;
  error_train_contour = 0;
  rnd_idx = randperm(length(imnames));
  for i = 1:length(imnames),
    name = imnames{rnd_idx(i)};
    im = imread(['../data/PASCAL/JPEGImages/' name '.jpg']);
    [mask] = imread(['../data/PASCAL/SegmentationObjectFilledDenseCRF/' name '.png']);
    [ims, masks] = sample_image(im, mask);
    ims = ims(:,:,[3,2,1],:);
    for c = 1:3, ims(:,:,c,:) = ims(:,:,c,:) - mean_pix(c); end
    ims = permute(ims,[2,1,3,4]);
    contours = zeros(size(masks),'single');
    for k = 1:8, contours(:,:,:,k) = imgradient(masks(:,:,:,k))>0; end
    contours = permute(contours,[2,1,3,4]);

    output = caffe('forward', {ims});  

    penalties = single(contours); penalties(contours==0) = 0.1; penalties = 10*penalties;
    [loss_contour, delta_contour] = loss_crossentropy_paired_sigmoid_grad(output{1}, contours, penalties);
    delta_contour = reshape(single(delta_contour),[H,W,1,8]);
    caffe('backward', {delta_contour});
    caffe('update');
    loss_train = loss_train + loss_contour;
    contours_pred = output{1} > 0;
    error_train_contour = error_train_contour + sum(sum(sum(contours_pred~=contours)));
  end
  error_train_contour  = error_train_contour / length(imnames);
  loss_train = loss_train / length(imnames);
  fprintf('Iter %d: training error is %f with contour in %f seconds.\n', iter, error_train_contour, toc);
  fprintf(fid, '%d %f\n', iter, error_train_contour);
  if mod(iter,5)==0, 
    weights = caffe('get_weights');
    save(sprintf('../results/PASCAL/%s-w10_model_iter%03d.mat', model_specs, iter), 'weights');
  end
end
fclose(fid);
caffe('snapshot');

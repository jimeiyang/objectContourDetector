% test semantic segmentation with vggnet fcn
addpath(genpath('../caffe-cedn/matlab'));
model_specs = sprintf('vgg-16-encoder-decoder-%s', 'contour');
use_gpu = true;
model_file = sprintf('%s.prototxt', model_specs);
solver_file = sprintf('%s_solver.prototxt', model_specs);
param = struct('base_lr', 0.00001, 'lr_policy', 'fixed', 'weight_decay', 0.001, 'solver_type', 3, 'snapshot_prefix', sprintf('../models/PASCAL/%s',model_specs));
make_solver_file(solver_file, model_file, param);
matcaffe_fcn_vgg_init(use_gpu, solver_file, 0);
caffe('set_phase_train');

iter = 5
load(sprintf('../models/contour/%s-w10-15-finetune-model-iter%03d.mat', model_specs, iter));
caffe('set_weights', weights);
caffe('snapshot');

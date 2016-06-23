function  init_matcaffe(solver_def_file, use_gpu, device_id)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1 || isempty(solver_def_file)
  error('You need the solver prototxt definition');
end

if nargin < 2 
  % By default use CPU
  use_gpu = 0;
end

caffe('init', solver_def_file);
fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
  caffe('set_device', device_id);
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

caffe('presolve');

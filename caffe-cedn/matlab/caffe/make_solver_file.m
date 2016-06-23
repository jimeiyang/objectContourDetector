function make_solver_file(solver_file, model_file, param)

param_default = struct('base_lr', 0.001, 'momentum', 0.9, 'weight_decay', 0.4, ...
    'lr_policy', 'step', 'gamma', 0.1, 'stepsize', 100000, 'display', 200, ...
    'max_iter', 600000, 'snapshot', 10000, 'snapshot_prefix', 'examples/model', 'solver_mode', 1, 'solver_type', 0);

if ~isfield(param, 'base_lr'), param.base_lr = param_default.base_lr; end
if ~isfield(param, 'momentum'), param.momentum = param_default.momentum; end
if ~isfield(param, 'weight_decay'), param.weight_decay = param_default.weight_decay; end
if ~isfield(param, 'lr_policy'), param.lr_policy = param_default.lr_policy; end
if ~isfield(param, 'gamma'), param.gamma = param_default.gamma; end
if ~isfield(param, 'stepsize'), param.stepsize = param_default.stepsize; end
if ~isfield(param, 'display'), param.display = param_default.display; end
if ~isfield(param, 'max_iter'), param.max_iter = param_default.max_iter; end
if ~isfield(param, 'snapshot'), param.snapshot = param_default.snapshot; end
if ~isfield(param, 'snapshot_prefix'), snapshot_prefix = model_file(1:strfind(model_file,'.')); 
    param.snapshot_prefix = snapshot_prefix; end
if ~isfield(param, 'solver_mode'), param.solver_mode = param_default.solver_mode; end
if ~isfield(param, 'solver_type'), param.solver_type = param_default.solver_type; end


fid = fopen(solver_file,'w');
fprintf(fid, '# The training protocol buffer definition\n');
fprintf(fid, 'train_net: "%s"\n',model_file);
fprintf(fid, '# The base learning rate, momentum and the weight decay of the network.\n');
fprintf(fid, 'base_lr: %f\n', param.base_lr);
if param.solver_type==0, fprintf(fid, 'momentum: %f\n', param.momentum); end
fprintf(fid, 'weight_decay: %f\n', param.weight_decay);
fprintf(fid, '# The learning rate policy\n');
fprintf(fid, 'lr_policy: "%s"\n', param.lr_policy);
fprintf(fid, 'gamma: %f\n', param.gamma);
fprintf(fid, 'stepsize: %d\n', param.stepsize);
fprintf(fid, '# Display every %d iterations\n', param.display);
fprintf(fid, 'display: %d\n', param.display);
fprintf(fid, '# The maximum number of iterations\n');
fprintf(fid, 'max_iter: %d\n', param.max_iter);
fprintf(fid, '# snapshot intermediate results\n');
fprintf(fid, 'snapshot: %d\n', param.snapshot);
fprintf(fid, 'snapshot_prefix: "%s"\n', param.snapshot_prefix);
fprintf(fid, '# solver mode: 0 for CPU and 1 for GPU\n');
fprintf(fid, 'solver_mode: %d\n', param.solver_mode);
fprintf(fid, 'solver_type: %d\n', param.solver_type);
fclose(fid);

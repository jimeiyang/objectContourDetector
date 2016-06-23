function make_model_file(model_file, input, layers)

fid = fopen(model_file, 'w');

fprintf(fid, '#--------input---------\n');
fprintf(fid, 'name: "%s"\n', input.name);
fprintf(fid, 'inputs: "%s"\n', input.inputs);
fprintf(fid, 'input_dim: %d\n', input.dim(1));
fprintf(fid, 'input_dim: %d\n', input.dim(2));
fprintf(fid, 'input_dim: %d\n', input.dim(3));
fprintf(fid, 'input_dim: %d\n', input.dim(4));

for i = 1 : length(layers)
    
end

fclose(fid);
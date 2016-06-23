% example
addpath(genpath('build'));

command = 'build/examples/dense_inference';
input_image = 'examples/im1.ppm';
input_anno = 'examples/anno1.ppm';
output = 'output1.ppm';
system(sprintf('%s %s %s %s', command, input_image, input_anno, output));
function [scores, maxlabel] = matcaffe_demo_mnist(use_gpu)

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init_mnist(use_gpu);
else
  matcaffe_init_mnist();
end


% prepare oversampled input
% input_data is Height x Width x Channel x Num
[train_x,train_y,test_x,test_y] = prepare_image_mnist();
% do forward pass to get scores
% scores are now Width x Height x Channels x Num
for n = 1:10
    caffe('set_phase_train');
    errors_train = 0;
    errors_test = 0;
    fprintf('Processing the %dth iteration.\n',n);
    for m = 1:6000
        %disp(m)
        input_data = train_x(:,:,(m-1)*10+1:m*10);
        scores = caffe('forward', {reshape(input_data,[28,28,1,10])});
        %fprintf('Done with forward pass.\n');

        scores = scores{1};
        %save scores_mnist.mat scores;
        [~,maxlabel] = max(scores,[],3);

        % you can also get network weights by calling

        layers = caffe('get_weights');
        % fprintf('Done with get weights\n');
        % caffe('set_weights',layers);
        % fprintf('Done with set weights\n');
%         save layers_minist.mat layers;

        y = train_y(:,(m-1)*10+1:m*10);
        y = reshape(y, [1,1,10,10]);
%         save('scores_mnist.mat','scores','y');

        [~,tlabel] = max(y,[],3);
        errors_train = errors_train + length(find(maxlabel ~= tlabel));

        delta = loss_euclidean_grad(scores, y);
%         save delta_mnist.mat delta
        f = caffe('backward',{delta});
        %fprintf('Done with backward\n');
        %d = caffe('get_all_diff');
        %fprintf('Done with get diff\n');
        caffe('update');
        %fprintf('Done with update\n');
        %save diff_cifar.mat d;
    end
    rate = errors_train / 60000 ;
    disp(rate)
    
    caffe('set_phase_test');
    for m = 1:1000
        input_data = test_x(:,:,(m-1)*10+1:m*10);
        scores = caffe('forward', {reshape(input_data,[28,28,1,10])});
        scores = scores{1};
        scores = squeeze(scores);
        [~,maxlabel] = max(scores);
        y = test_y(:,(m-1)*10+1:m*10);
        [~,tlabel] = max(y);
        errors_test = errors_test + length(find(maxlabel ~= tlabel));
    end
    rate = errors_test / 10000 ;
    disp(rate)
end

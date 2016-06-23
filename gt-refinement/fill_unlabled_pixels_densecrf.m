% fill the unlabeled pixels with graphcut
addpath(genpath('densecrf/build'));
names = textread('../data/PASCAL/train.txt','%s');

mkdir('../data/PASCAL/SegmentationObjectFilledDenseCRF');
mkdir('../data/PASCAL/ContourObject');
for i = 1:length(names)
    im = imread(sprintf('../data/PASCAL/JPEGImages/%s.jpg',names{i}));
    G = fspecial('gaussian',[5 5],2);
    im = imfilter(im,G,'same');
    [h,w,~] = size(im);
    [input,cmap] = imread(sprintf('../data/PASCAL/SegmentationObject/%s.png',names{i}));
    input(input==0) = 254;
    input(input==255) = 0;
    input = input+1;
    input = cmap(input(:),:);
    input = permute(reshape(input',[3,h,w]),[2,3,1]);
    imwrite(uint8(255*input),'input_anno.ppm','ppm');
    imwrite(im,'input_image.ppm','ppm');
    command = 'densecrf/build/examples/dense_inference';
    input_image = 'input_image.ppm';
    input_anno = 'input_anno.ppm';
    output = 'output.ppm';
    system(sprintf('%s %s %s %s', command, input_image, input_anno, output));
    output = double(imread('output.ppm'));
    colorcode = cmap(:,1)*255+cmap(:,2)*255*255+cmap(:,3)*255*255*255;
    outputcode = output(:,:,1)+output(:,:,2)*255+output(:,:,3)*255*255;
    codes = unique(outputcode);
    label = zeros(h,w,'uint8');
    for j=1:length(codes),
        id = find(colorcode==codes(j));
        label(outputcode==codes(j)) = id-1;
    end
    label(label==254) = 0;
    contour = imgradient(label);
    imwrite(label,cmap,sprintf('../data/PASCAL/SegmentationObjectFilledDenseCRF/%s.png',names{i}),'png');
    imwrite(contour>0,sprintf('../data/PASCAL/ContourObject/%s.png',names{i}),'png');
    if mod(i,100)==0, disp(i); end
%     figure(1);imshow(label,cmap);
%     figure(2);imshow(im2double(im)+cat(3,contour,contour,contour));
%     pause;
end

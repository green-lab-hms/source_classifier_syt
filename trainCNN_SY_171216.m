clear,
manLabels.labels = [];
manLabels.sources = [];

%% Load in Data
path = 'Z:\HarveyLab\Tier1\Shih_Yi\Imaging\classified_sessions\second_batch';
list  = ls([path,filesep,'*.mat*']);

for iFile = 1 : size(list,1)
    file = fullfile(path, strtrim(list(iFile,:)));
    load(file);
    
    includeID = find(labels<6);
    labels = labels(includeID);
    sources = sources(:,includeID);

    manLabels.labels = cat(1,manLabels.labels,labels(:));
    manLabels.sources = cat(2,manLabels.sources,sources);
end

length(manLabels.labels),

%% Align and assign category

% Normalize Sources
manLabels.sources = bsxfun(@rdivide,...
    manLabels.sources,sqrt(sum(manLabels.sources.^2)));

% Align Sources
winRad = 12;
winWidth = 2*winRad+1;
nROIs = size(manLabels.sources,2);
alignedMasks = zeros(winWidth,winWidth,nROIs);
parfor nROI = 1:nROIs
    thisMask = padarray(reshape(manLabels.sources(:,nROI),512,512),[winRad,winRad],0);
    maskProps = regionprops(thisMask>0, thisMask, 'WeightedCentroid','Area');
    if length(maskProps)>1
        [~,thisComp] = max([maskProps.Area]);
        maskProps = maskProps(thisComp);
    end
    thisCent = round(maskProps.WeightedCentroid([2,1]));
    tempMask = thisMask(thisCent(1)-winRad:thisCent(1)+winRad,...
        thisCent(2)-winRad:thisCent(2)+winRad);
    alignedMasks(:,:,nROI) = tempMask;
end
imgSize = [winWidth winWidth];

%% Merge old training set with new one
oldData = load('alignedData_new2_171216.mat');
realLabels = cat(1, oldData.realLabels_new2, manLabels.labels);
alignedMasks = cat(3, oldData.alignedMasks, alignedMasks);

%% Augment data:

augmentedMasks = alignedMasks;
nClasses = numel(unique(realLabels));
classSize = histcounts(realLabels);
classSizeReal = classSize;
% nRealExamples = sum(~isnan(sourceClass));
nRealExamples = length(realLabels);

realInd = {};
for i = 1:nClasses
    realInd{i} = find(realLabels==i);
end

% Determine needed class types and indeces and pre-allocate:
% Equalizing number of each class
nFoldAug = 12;
nTotal = size(alignedMasks,3)*nFoldAug;
augmentedClass = realLabels;
augmentedInd = (1:length(realLabels))';
nPerClass = round(nTotal/nClasses);
nTotal = nPerClass*nClasses;
augmentedMasks(:,:,nTotal) = 0;
augPerClass = bsxfun(@minus,nPerClass,classSizeReal);

for iClass = 1:nClasses
    augmentedClass = cat(1,augmentedClass,ones(augPerClass(iClass),1)*iClass);
    augmentedInd = cat(1,augmentedInd,realInd{iClass}(randi(classSizeReal(iClass),augPerClass(iClass),1)));
end


%% 
tic;
% Create transformed masks
tformTemplate = affine2d(eye(3));
maxTranslPix = 3;
maxScaleFac = 0.2;
outRef = imref2d(size(augmentedMasks(:,:,1)));
outRef.XWorldLimits = outRef.XWorldLimits-imgSize(1)/2-0.5;
outRef.YWorldLimits = outRef.YWorldLimits-imgSize(2)/2-0.5;

augmentedMasks = alignedMasks(:,:,augmentedInd);
parfor_progress(nTotal);
parfor i = (nRealExamples + 1):nTotal
    parfor_progress;
    
    % Pick real example
%     classHere = augmentedClass(i);
%     realIndHere = augmentedInd(i);
%     realIndHere = realInd{classHere}(randi(classSizeReal(classHere)));    
    thisMask = augmentedMasks(:,:,i);

    % Create random rotation, translation, and scaling:
%     augmentedClass(i) = realLabels(realIndHere);
    x = (rand*2-1) * maxTranslPix;
    y = (rand*2-1) * maxTranslPix;
    scFac = (rand*2-1) * maxScaleFac + 1;
     
    th = rand * 360;
    
    % Matrix representing a rotation followed by scaling followed by a translation:
    % http://planning.cs.uiuc.edu/node99.html
    tform = tformTemplate;
    tform.T = [cosd(th) -sind(th) x;
        sind(th)  cosd(th) y;
        0         0 1]';
    tform.T(1:2,1:2) = tform.T(1:2,1:2) .* scFac;
    
    augmentedMasks(:,:,i) = imwarp(thisMask, ...
        outRef, tform, 'FillValues', 0, ...
        'OutputView', outRef);
end
parfor_progress(0);
toc;

%%
save('C:\Users\Shih-Yi\Documents\MATLAB\CNN_classifySources\trainingData_171216.mat',...
    'alignedMasks', 'realLabels','augmentedMasks', 'augmentedClass','augmentedInd','-v7.3');

%% Adjust data format:
clear X Y;

% Input for NN, with dimensions h-w-ch-n:
X = permute(augmentedMasks, [1 2 4 3]);
Y = categorical(augmentedClass);
nCat = numel(unique(Y));
classSizes = histcounts(Y);

% Randomize order:
ord = randperm(numel(Y));
Y = Y(ord);
X = X(:,:,:,ord);

fprintf('Class sizes: ')
fprintf('%d ', histcounts(Y));
fprintf('\n')

%% Matthias' Neural network (modified)
layers = [imageInputLayer(imgSize,'Normalization','zerocenter','Name','inputl','DataAugmentation','randfliplr')
    convolution2dLayer(5, 32, 'Stride', 1, 'Name', 'conv1', 'Padding', 0)
    reluLayer('Name','relu1')
    convolution2dLayer(5, 16, 'Stride', 1, 'Name', 'conv2', 'Padding', 0)
    reluLayer('Name','relu2')
    convolution2dLayer(5, 16, 'Stride', 1, 'Name', 'conv3', 'Padding', 0)
    reluLayer('Name','relu3')
    fullyConnectedLayer(256, 'Name', 'full1')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(nCat, 'Name', 'full2')
    softmaxLayer('Name', 'softm')
    classificationLayer('Name', 'out')];     

%% Train:
clear options;

options = trainingOptions('sgdm','MaxEpochs',30,...
    'MiniBatchSize', 1e3, ...
    'L2Regularization', 0.0001, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 1, ...
    'LearnRateDropPeriod', 1);

nSamples = numel(Y);
convnet = trainNetwork(X(:,:,:,1:nSamples), Y(1:nSamples), layers, options);

save('C:\Users\Shih-Yi\Documents\MATLAB\CNN_classifySources\convNet_unrefined_171216.mat', 'convnet');

%% Refinement:
options = trainingOptions('sgdm','MaxEpochs',40,...
    'MiniBatchSize', 1e3, ...
    'L2Regularization', 0.0001, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.4, ...
    'LearnRateDropPeriod', 5,...
    'CheckpointPath', 'C:\Users\Shih-Yi\Documents\MATLAB\CNN_classifySources\CNN_171216_CheckPoints');

nSamples = numel(Y);
convnet = trainNetwork(X(:,:,:,1:nSamples), Y(1:nSamples), convnet.Layers, options);

save('C:\Users\Shih-Yi\Documents\MATLAB\CNN_classifySources\convNet_171216.mat', 'convnet');

%% Confusion matrix:
net = convnet;
Xtest = permute(alignedMasks, [1 2 4 3]);
[YTest, score] = classify(net, Xtest);

nY = numel(realLabels);
targets = false(nCat, nY);
targets(realLabels' + (0:(nY-1))*nCat) = true;

outputs = false(nCat, nY);
outputs(double(YTest)' + (0:(nY-1))*nCat) = true;

figure;plotconfusion(double(targets), double(outputs))

%% Visualize mis-classified cases
Y_cnn = double(YTest);

T1O2ind = find(realLabels == 1 & Y_cnn == 2);
T1O2 = Xtest(:,:,:,T1O2ind);
figure, montage(T1O2*2);
title('Target 1, Output 2');

T1O3ind = find(realLabels == 1 & Y_cnn == 3);
T1O3 = Xtest(:,:,:,T1O3ind);
figure, montage(T1O3*2);
title('Target 1, Output 3');

T1O4ind = find(realLabels == 1 & Y_cnn == 4);
T1O4 = Xtest(:,:,:,T1O4ind);
figure, montage(T1O4*2);
title('Target 1, Output 4');

T2O1ind = find(realLabels == 2 & Y_cnn == 1);
T2O1 = Xtest(:,:,:,T2O1ind);
figure, montage(T2O1*2);
title('Target 2, Output 1');

T3O1ind = find(realLabels == 3 & Y_cnn == 1);
T3O1 = Xtest(:,:,:,T3O1ind);
figure, montage(T3O1*2);
title('Target 3, Output 1');

T4O1ind = find(realLabels == 4 & Y_cnn == 1);
T4O1 = Xtest(:,:,:,T4O1ind);
figure, montage(T4O1*2);
title('Target 4, Output 1');

T1O1ind = find(realLabels == 1 & Y_cnn == 1);
T1O1 = Xtest(:,:,:,T1O1ind(1:256));
figure, montage(T1O1*2);
title('Target 1, Output 1');

T2O2ind = find(realLabels == 2 & Y_cnn == 2);
T2O2 = Xtest(:,:,:,T2O2ind(1:256));
figure, montage(T2O2*2);
title('Target 2, Output 2');

T3O3ind = find(realLabels == 3 & Y_cnn == 3);
T3O3 = Xtest(:,:,:,T3O3ind(1:256));
figure, montage(T3O3*2);
title('Target 3, Output 3');

T4O4ind = find(realLabels == 4 & Y_cnn == 4);
T4O4 = Xtest(:,:,:,T4O4ind(1:256));
figure, montage(T4O4*2);
title('Target 4, Output 4');


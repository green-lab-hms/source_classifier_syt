%% Load Data
clear, close all
load('Z:\HarveyLab\Tier1\Shih_Yi\Imaging\9\170819\Slice01_patchResults_v170311.mat','A','C');
% load('Z:\HarveyLab\Tier1\Shih_Yi\Imaging\9\170819\Slice01_fullResTraces_v170311.mat','Cf');
% memMap = matfile('Z:\HarveyLab\Tier1\Shih_Yi\Imaging\7\170719\Slice3_MemMapReconstructed.mat');
% Y = memMap.Y(:,:,1000:2000);

%% Prepare GUI and Labels

% A = A ./ sqrt(sum(A.^2));
% 
% winRad = 12;
% winWidth = 2*winRad+1;
% nROIs = size(A,2);
% alignedMasks = zeros(winWidth,winWidth,nROIs);
% allCentroids = nan(nROIs,2);
% % allSkews = nan(nROIs,1);
% for nROI = 1:nROIs
%     thisMask = padarray(reshape(A(:,nROI),512,512),[winRad,winRad],0);
%     maskProps = regionprops(thisMask>0, thisMask, 'WeightedCentroid','Area');
%     if length(maskProps)>1
%         [~,thisComp] = max([maskProps.Area]);
%         maskProps = maskProps(thisComp);
%     end
%     allCentroids(nROI,:) = maskProps.WeightedCentroid;
% end

% l = clusterSourcesWithCurrentNn(A);
% o = classifierGui(A,l,allCentroids);
l = clusterSourcesWithCurrentNn(A,'convNet_l23_171216.mat');
o = classifierGui_SY(A,l);

%% interact with GUI
% change displayed class, for example
setCurrentClass(o,2);

% click on sources that are mis-classified, and their labels will be set to nan

% reclassify nans to sepcific category, for example
nan2label(o, 4);

% after reclassifying sources, find new labels in o.labels 

%% display all spatial maps, movies, and initial histogram
allSpatial = reshape(sum(o.sources,2),512,512);
figure(1),imagesc(allSpatial,[0 0.5])
implay(Y/2e3);
figure(5),clf;hold on; histogram(o.labels);

%% visualize spatial map and trace of one (and only one) chosen patch
ind = find(isnan(o.labels));
figure(2),clf;hold on; plot(Cf(ind,:))
figure(3),clf;hold on; imagesc(reshape(A(:,ind),512,512));hold on;
axis ij;axis equal

%% crop the location of the chosen patch on the first frame of the movie
figure(2);
clf;hold on;
imshow(imNorm(Y(:,:,1))); colormap(gray);hold on;
centroid = o.centroids(ind,:);
h = plot(centroid(1),centroid(2), 'ro');

%% save file
fileName = 'Z:\HarveyLab\Tier1\Shih_Yi\Imaging\classified_sessions\layer5\second_batch\5_170716_V1l5_Slice3.mat';
figure(5),clf;hold on; histogram(o.labels);
saveData(o, fileName);

classdef classifierGui_SY < handle
    
    properties
        sources
        labels
        classNames
        centroids
        currentClass
        hFig
        hImg
        hAx
        overlayImg % An optional image to overlay the sources, e.g. for red label in inhibitory neurons.
    end
    
    methods
        function o = classifierGui_SY(sources, labels)
            
            sources = sources ./ sqrt(sum(sources.^2));

            winRad = 12;
            winWidth = 2*winRad+1;
            nROIs = size(sources,2);
            alignedMasks = zeros(winWidth,winWidth,nROIs);
            allCentroids = nan(nROIs,2);

            for nROI = 1:nROIs
                thisMask = padarray(reshape(sources(:,nROI),512,512),[winRad,winRad],0);
                maskProps = regionprops(thisMask>0, thisMask, 'WeightedCentroid','Area');
                if length(maskProps)>1
                    [~,thisComp] = max([maskProps.Area]);
                    maskProps = maskProps(thisComp);
                end
                allCentroids(nROI,:) = maskProps.WeightedCentroid;
            end

            o.sources = sources;
            o.labels = labels;
            o.centroids = allCentroids-(winRad+0.5); 
            o.currentClass = 1;
            o.drawFig;
            o.classNames = {'Soma', ...
                'smallRoundProcess', 'complexProcess', ...
                'junkOrGrabBag', 'possibleCell'};
            
            if any(isnan(o.labels))
                nan2newLabel(o, 'originalNans')
            end
        end
        
        function drawFig(o)
            o.hFig = figure(234234);
            clf

            if isempty(o.overlayImg)
                o.hImg = imshow(2*mat2gray(full(...
                    reshape(sum(o.sources(:,o.labels==o.currentClass), 2), 512, 512))), ...
                    'InitialMagnification', 'fit');
            else
                cIm = zeros(512,512,3);
                cIm(:,:,1) = o.overlayImg;
                cIm(:,:,2) = 2*mat2gray(full(...
                    reshape(sum(o.sources(:,o.labels==o.currentClass), 2), 512, 512)));
                o.hImg = imshow(cIm, 'InitialMagnification', 'fit');
            end
            
            o.hAx = gca;
            
%             cIm = zeros(512,512,3);
%             nc = 100;
%             clrs = hsv(nc);
%             sel = o.labels==o.currentClass;
% 
%             % Randomize reproducibly:
%             ord = o.centroids(:, 1)-floor(o.centroids(:, 1));
%             [~, ord] = sort(ord);
% 
%             for i = find(sel(:))'
%     
%                 ic = mod(ord(i), nc);
%     
%                 if ic==0
%                     ic = nc;
%                 end
%                 aHere = mat2gray(full(reshape(o.sources(:,i), 512, 512)));
%                 cIm = cIm + bsxfun(@times, aHere, permute(clrs(ic, :), [1, 3, 2]));
%             end
%             o.hImg = imshow(cIm, 'initialMagnification', 'fit');
            
            o.hImg.ButtonDownFcn = @o.cbRemoveSource;
%             hold on
%             isDisplayed = o.labels == o.currentClass;
%             plot(o.centroids(isDisplayed, 1)-12.5, o.centroids(isDisplayed, 2)-12.5, '.')
            if numel(o.classNames) >= o.currentClass
                title(o.classNames{o.currentClass});
            end
        end
        
        function setCurrentClass(o, c)
            o.currentClass = c;
            o.drawFig;
        end
        
        function cbRemoveSource(o, ~, ~)
            isDisplayed = o.labels == o.currentClass;
%             coords = abs(bsxfun(@minus, o.centroids, o.hAx.CurrentPoint(1, [2, 1])));
            coords = abs(bsxfun(@minus, o.centroids, o.hAx.CurrentPoint(1, [1,2])));
            dist = hypot(coords(:, 1), coords(:, 2));
            dist(~isDisplayed) = inf;
            [~, i] = min(dist);
            o.labels(i) = nan;
            o.drawFig;
        end
        
        function nan2newLabel(o, newLabelName)
            o.labels(isnan(o.labels)) = max(o.labels)+1;
            o.classNames{end+1} = newLabelName;
        end
        
        function nan2label(o, label)
            o.labels(isnan(o.labels)) = label;
        end
        
        function saveData(o, fileName)
            sources = o.sources;
            labels = o.labels;
            centroids = o.centroids;
            classNames = o.classNames;
            save(fileName, 'sources', 'labels', 'centroids', 'classNames', '-v7.3')
        end
    end
end

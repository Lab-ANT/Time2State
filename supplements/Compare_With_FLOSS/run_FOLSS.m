% Created by Chengyu on 2023/4/20.
% This is a script for running segmentation on all datasets,
% except for the UCR-SEG dataset, UCR-SEG has a script implemented
% by the authors.

% Different from The original code, the positions of cut points ( or segment boundaries)
% are saved in files ended with .cp, in case that some file names will
% exceed the maximum length of limitation. Some data in the datasets may
% have many cut points.

% stepSize is set to the dafault value used in flossScore.m
stepSize = 3; 

% params: dataset, stepSize, subSequenceLength (i.e., slWindow)
% run_on('MoCap', 3, 100);
% run_on('ActRecTut', 3 , 80);
% run_on('synthetic_data', 3, 50);
% run_on('USC-HAD', 3, 40);
% run_on('PAMAP2', 3, 16);

% run_on('MoCap', 3, 100);
% run_on('ActRecTut', 3 , 16);
% run_on('synthetic_data', 3, 10);
% run_on('USC-HAD', 3, 16);
run_on('PAMAP2', 3, 16);

function [] = run_on(dataset, stepSize, subSequnceLength)
    % create path for saving the crosscount;
    % corsscount is only saved for displaying usage.
    mkdir(['output_FLOSS\crosscount\',dataset]);
    path = ['data\FLOSS_format\',dataset,'\*.txt'];
    cps_path = ['data\FLOSS_format\',dataset,'\*.cp'];
    fileID = fopen(['output_FLOSS\', dataset, '_segpos.txt'],'w');
    [vals, fnames, numfids] = readFiles(path);
    [vals_cp, fnames_cp, numfids_cp] = read_CP_Files(cps_path);
    
    
    for filesInd = 1:numfids
        %  fileID2 = fopen(['output_FLOSS\crosscount\',dataset,'\',fnames{filesInd}]);
        t = 0;
        tic;
        groundTruthSegPos = vals_cp{filesInd}';
        disp('Working on....');
        disp(fnames(filesInd,:));
        data = vals{filesInd};
        [averaged_crosscount] = SegmentMultivariateTimeSeries(data, subSequnceLength);
        [~, n] = size(groundTruthSegPos);
        [localMinimums, indLM] = findLocalMinimums(averaged_crosscount, stepSize*subSequnceLength, n);
        [~, dataLength] = size(averaged_crosscount);
        score = calcScore(groundTruthSegPos, indLM, dataLength);
        write2File(fileID,fnames(filesInd,:),indLM, score);
        writematrix(averaged_crosscount', ['output_FLOSS\crosscount\',dataset,'\',fnames(filesInd,:)]);
        t = t + toc;
        t
        % dlmwrite(['output_FLOSS\crosscount\',dataset,'\',fnames(filesInd,:)], averaged_crosscount');
    end
    fclose(fileID);
end

% This function is used to read groundtruth files.
function  [vals, files, numfids] = read_CP_Files(path)
path = strrep(path, '*.cp', '');
files1 = dir(strcat(path,'*.cp'));
files = strvcat( files1.name );
[numfids, ~] = size(files);

vals = cell(1,numfids);
for filesInd = 1:numfids
    vals{filesInd} = importdata(strcat(path,files(filesInd,:)));
end
end

% The following part is not modified.
function [minV, ind]= findLocalMinimums(data, length, n)
%% length
%% n the number of minimum
minV(1:n) = inf;
ind(1:n) = -1;
for i=1:n
    [minV(i), ind(i)] = min(data);
    data(ind(i)-length:ind(i)+length) = inf;
end
end

function score = calcScore(groundTruth, detectedSegLoc, dataLength)
[~, n] = size(groundTruth);
[~, k] = size(detectedSegLoc);
ind(1:n) = -1;
minV(1:n) = inf;

for j = 1:1:n
    for i = 1:1:k
        if(abs(detectedSegLoc(i)-groundTruth(j)) < abs(minV(j)))
            minV(j) = abs(detectedSegLoc(i) - groundTruth(j));
            ind(j) = i;
        end
    end
end

sumOfDiff = sum(minV);
score = sumOfDiff/dataLength;
end

%% read data from all files
function  [vals, files, numfids] = readFiles(path)
path = strrep(path, '*.txt', '');
files1 = dir(strcat(path,'*.txt'));
files = strvcat( files1.name );
[numfids, ~] = size(files);

vals = cell(1,numfids);
for filesInd = 1:numfids
    vals{filesInd} = importdata(strcat(path,files(filesInd,:)));
end
end

function [groundTruthSegPos, length] = getSegmentPos(names)
segmentPos = strfind(names,'_');
[~, n] = size(segmentPos);
data = [ 0 0];
for i = 1:1:n
    if(i+1 <= n)
        data(i) = str2num(names(segmentPos(i)+1:(segmentPos(i+1)-1)));
    else
        endPos = strfind(names,'.txt');
        data(i) = str2num(names(segmentPos(i) + 1:(endPos-1))); 
    end
end
length = data(1);
groundTruthSegPos =  data(2:end);
end

%% write false result in file for test
function write2File(fileID, name, predictedSegment, score)
fprintf(fileID,name);
fprintf(fileID,' , ');
[~, n]= size(predictedSegment);
for i=1:1:n
    fprintf(fileID,num2str(predictedSegment(i)));
    if(i~=n)
        fprintf(fileID,'_');
    else
        fprintf(fileID,',');
    end
end
fprintf(fileID,num2str(score));
fprintf(fileID,'\n');
end
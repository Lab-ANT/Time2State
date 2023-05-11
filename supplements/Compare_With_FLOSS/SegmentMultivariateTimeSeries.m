% Created by Chengyu on 2023/4/20.
% Modified version of FLOSS/FLUSS for segmenting multivariate time series;
% According to the original paper, the overall crosscount is calculated by
% averaging the crosscount curves of all channels.
% 1. All channel shares the same slWindow.
% 2. The crosscount for each channel is calculated by RunSegmentation function
% provided in the latestCode/.

function [averaged_crosscount] = ...
    SegmentMultivariateTimeSeries(ts, slWindow )

    % Get total length and #channels.
    [nrows, nchannels] = size(ts);

    % To save crosscount for each channels and calculate averaged
    % crosscount; length = nrows-slWindow.
    averaged_crosscount = zeros(1, nrows-slWindow);

    for channelInd = 1:nchannels
        disp('channel')
        channelData = ts(:,channelInd);
        [crosscount] = RunSegmentation(channelData, slWindow);
        averaged_crosscount = averaged_crosscount+crosscount;
    end

    [averaged_crosscount] = averaged_crosscount/nchannels;
    
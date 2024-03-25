function [ranked_features] = iter_ranking(data, group)

%%
% data has m samples with n features
% group contains the class labels for data
% The function returns features indices in order of their ranking
%% 

[m,n] = size(data)

for i = 1:n
    [err,thresh] = majorityvotefornumbers(data(:,i),group);
    fea_err(i) = sum(err);
end

[val, iter_fea_ind] = sort(fea_err); 
ranked_features = iter_fea_ind';
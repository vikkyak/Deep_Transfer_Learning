function [F] = featureSelection_IterRanking(trainData,trainLabels,k,nFeatureSelect)

	[~,nFeatures] = size(trainData{1});
	
	% Performing Iter_Ranking on each fold after k-fold cross validation	
	for i=1:k
      		tRank{i} = tempParallelRank(i,trainData,trainLabels);
 	end

	% Summing of rank of each feature after k-fold cross validation
 	for i=1:nFeatures
 		rankSumofEachFeature(i)=0;
 		for j=1:k
 			rankSumofEachFeature(i) = rankSumofEachFeature(i) + find(tRank{j}==i);
 		end 
 	end
 
	
 	featureRank = sort(rankSumofEachFeature);
 	
	%Selecting first nFeatureSelect out of nFeatures
	ind = 0;
	for i=1:nFeatureSelect
    		temp = find(rankSumofEachFeature==featureRank(i));
    		for j=1:length(temp)
        		ind = ind + 1;
        		F(ind)=temp(j);  
    		end;
	end;
end

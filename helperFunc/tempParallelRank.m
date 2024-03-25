function [tRank] = tempParallelRank(i,trainData,trainLabels)

      % Zero-One Normalization of data
      minm = min(min(trainData{i}));
      maxm = max(max(trainData{i}));
      trainData{i} = (trainData{i} - minm*(ones(size(trainData{i}))))./(maxm - minm);

	
      tRank = iter_ranking(trainData{i},trainLabels{i});
	
end

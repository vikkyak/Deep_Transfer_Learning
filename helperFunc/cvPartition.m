function [trainData,trainLabels,testData,testLabels] = cvPartition(labData,k)
	labeled_data = labData(:,1:end-1);
	labeled_group = labData(:,end);

	c = cvpartition(labeled_group,'kfold',k);

	for i = 1:k
    		trainData{i} = labeled_data(training(c,i),:);
    		trainLabels{i} = labeled_group(training(c,i),:);
    		testData{i} = labeled_data(test(c,i),:);
    		testLabels{i} = labeled_group(test(c,i),:);
	end

end


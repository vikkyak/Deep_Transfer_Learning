function [auc_soft] = softmaxClassifier(trainData,trainLabels,testData,testLabels,numClasses)
	
	[nFeatures,nSamples] = size(trainData);
	softmaxTheta = 0.005 * randn(nFeatures*numClasses,1);
 
	% used softmaxTrain of minFunc,
	% set saeSoftmaxOptTheta = softmaxModel.optTheta(:);
	softmaxModel = struct; 
	options.maxIter = 200;
	lambda = 1e-4;
	softmaxModel = softmaxTrain(nFeatures, numClasses, lambda, ...
			trainData, trainLabels,options);
	softmaxOptTheta = softmaxModel.optTheta(:);

	% Reshape the softmaxOptTheta
	softmaxOptTheta = reshape(softmaxOptTheta(1:nFeatures*numClasses), numClasses, nFeatures);

	M = softmaxOptTheta * testData;
	M = bsxfun(@minus, M, max(M));
	p = bsxfun(@rdivide, exp(M), sum(exp(M)));
	[Max, pred] = max(log(p));

	[~,~,T,auc_soft] = perfcurve(testLabels,pred(1,:),1);
	acc = mean(testLabels(:) == pred(:));
end

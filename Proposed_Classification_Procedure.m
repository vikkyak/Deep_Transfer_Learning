function Proposed_Classification_Procedure()
	
	clear all;
	clc;
	
	addpath('libsvm');
	addpath('minFunc');
	addpath('helperFunc');
	%% Partition of labelled Data(labData) and unlabelled Data (unlabData)

	% Suppose classication is permorfed between Colon & Uterus then...
	% Colon_Uterus dataset is taken to be labData while Breast, Kidney,... 
	% Lung, Omentum, Prostate, endometrium and Ovary are taken to be unlabData.

    % Loading Data and partioning into labelled and unlabelled data
	[labData,unlabData] = partition();    % data is matrix where each row represnt 
					          % a sample while each column represent a feature

	% divide labelled data into train and test datasets using cvpartition function of matlab
	k = 10;               % 10 - fold cross validation is used here
	[trainData,trainLabels,testData,testLabels] = cvPartition(labData, k);   
	
	nFeatureSelect = 1024;                                         % Number of features to be selected is 1024
	F = featureSelection_IterRanking(trainData,trainLabels,k,nFeatureSelect);  % perform feature selection with ITER ranking 	
								       % F is first best nFeatureSelect 
	
    % F is first best nFeatureSelect 
	
	unlabData = unlabData(:,F);      % choosing only the best F features
	
	[normUnlabData, min_of_all, max_of_all] = Zero_One_Normalization(unlabData);    % Normalization of unlabData
	
	numClasses = 2;
	inputSize  = 1024;     % InputLayer Size
	hiddenSizeL1 = 900;    % Layer 1 Hidden Size
	hiddenSizeL2 = 800;    % Layer 2 Hidden Size
	hiddenSizeL3 = 750;    % Layer 3 Hidden Size
	
	networkConfig = [inputSize,hiddenSizeL1,hiddenSizeL2,hiddenSizeL3];

	lambda = 3e-3;          % weight decay parameter
	beta = 3;               % weight of sparsity penalty term
	sparsityParam = 0.1;  % desired average activation of the hidden units.
			      % sprsityParam varies from [0.1,0.2,0.3,....,0.9] and reported the maximum AUC among all.  
	param = [lambda,beta,sparsityParam];

	initializedWeights = stacked_autoencoder(normUnlabData',networkConfig,param);
	
	for i = 1:k
		train_data = trainData{i}(:,F)';            % train_data is a matrix with each row represent a feature
		train_labels = trainLabels{i}';		    % and each column represent a sample
 		train_labels(train_labels==0) = 2;
		
		% Normalization of trainData with parameters of unlabData
		[normTrainData,~,~] = Zero_One_Normalization(train_data,min_of_all,max_of_all); 

		% Finetune weights using median based finetune
        	theta = cell2mat(initializedWeights);
		[finetuned_stack, clust_param] = median_finetune(theta,networkConfig,normTrainData,train_labels);

		% Generate high level features for train data with the finetuned stacked autoencoder
		trainFeatures = stackedAE_Out(normTrainData,finetuned_stack);

		% Generate test Features         
		test_data = testData{i}(:,F)';            % test_data is a matrix with each row represent a feature
		test_labels = testLabels{i}';		    % and each column represent a sample
 		test_labels(test_labels==0) = 2;
		
		% Normalization of testData with parameters of unlabData
		[normTestData,~,~] = Zero_One_Normalization(test_data,min_of_all,max_of_all); 
		testFeatures = stackedAE_Out(normTestData,finetuned_stack);

		%train classifier on high level features
		% We will be using various classifiers
		% --> SVM with parameter tuned
		% --> Random Forest
		% --> Softmax Classifier
		
		[auc_SVM_lin(i),auc_SVM_rbf(i)] = SVMclassifier(trainFeatures',train_labels,testFeatures',test_labels); 
		[auc_RF(i)] = RFclassifier(trainFeatures',train_labels,testFeatures',test_labels); 
		[auc_Soft(i)] = softmaxClassifier(trainFeatures,train_labels,testFeatures,test_labels,numClasses); 

	end
	 
	
	fprintf('After Finetuning Test AUC using softMax classifier: %0.3f%%\n', mean(auc_Soft)*100);
	fprintf('After Finetuning Test AUC using linear SVM classifier: %0.3f%%\n', mean(auc_SVM_lin));
	fprintf('After Finetuning Test AUC using rbf SVM classifier: %0.3f%%\n', mean(auc_SVM_rbf));
	fprintf('After Finetuning Test AUC using random forest classifier: %0.3f%%\n', mean(auc_RF)*100);
	
	
end

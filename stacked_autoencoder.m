function [initializedWeights] = stacked_autoencoder(data,networkConfig,param)
	
	% network architecture with L hidden layers
	L = length(networkConfig)-1;
	inputSize  = networkConfig(1);
	hiddenSizeL = [];     
	for i = 1:L
		hiddenSizeL(i) = networkConfig(i+1);
	end

    % creating a new variable inputData as part of learning autoencoders
	inputData = data; 

	initializedWeights = cell(L,1);
	
	for i=1:L		
		% learn the weights and bias (i-1)th and ith hidden layer with sparse autoencoder and then
		% perform feedforward operation until ith hidden layer to obtain the activation outputs which would be used to train the next autoencoder  		
		if(i==1)
			initializedWeights{i} = sparse_autoencoder(inputSize,hiddenSizeL(i),inputData,param);
			inputData = feed_forward(initializedWeights{i},hiddenSizeL(i),inputSize,inputData);
		else
			initializedWeights{i} = sparse_autoencoder(hiddenSizeL(i-1),hiddenSizeL(i),inputData,param);
			inputData = feed_forward(initializedWeights{i},hiddenSizeL(i),hiddenSizeL(i-1),inputData);
		end

		
	end

end

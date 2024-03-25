function [optWeights] = sparse_autoencoder(inputSize,hiddenSize,data,param)
	
	
	lambda = param(1);              % lambda: weight decay parameter
	beta = param(2);                % beta: weight of sparsity penalty term
	sparsityParam = param(3);       % sparsityParam: The desired average activation for the hidden units 

	desiredOut = data;   % assigning desired output to be the same as the input unlabelled data 
	
	%  Obtain random parameters weights and bias
	randomWeights = initializeParameters(hiddenSize,inputSize);

	% Defines the cost function J(W,b)
	cost = sparseAutoencoderCost(randomWeights,inputSize,hiddenSize,data,lambda,sparsityParam,beta); 

	%  Use minFunc to minimize the function
	addpath minFunc/
	options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
        	                  % function. Generally, for minFunc to work, you
        	                  % need a function pointer with two outputs: the
        	                  % function value and the gradient. In our problem,
        	                  % sparseAutoencoderCost.m satisfies this.
	options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
	options.display = 'on';

	[optWeights, cost] = minFunc( @(p) back_propagation(p, ...
					inputSize, hiddenSize, ...
					lambda, sparsityParam, ...
					beta, data), ...
					randomWeights, options);

	

end

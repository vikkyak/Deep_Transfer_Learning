function [cost] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
					     lambda, sparsityParam, beta,data)
 
	% visibleSize: the number of input units 
	% hiddenSize: the number of hidden units 
	% lambda: weight decay parameter
	% sparsityParam: The desired average activation for the hidden units 
	% beta: weight of sparsity penalty term

	% The input theta is a vector (because minFunc expects the parameters to be a vector). 
	% We first convert theta to the (W1, W2, b1, b2) matrix/vector format
 
	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
 
	
	% Here, we initialize cost to zero. 
	cost = 0;

	%Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder

 
	[nFeatures, nSamples] = size(data);
	% first calculate the regular cost function J
 
	[a1, a2, a3] = getActivation(W1, W2, b1, b2, data);
	errtp = ((a3 - data) .^ 2) ./ 2;
	err = sum(sum(errtp)) ./ nSamples;
	% now calculate pj which is the average activation of hidden units
	pj = sum(a2, 2) ./ nSamples;
	% the second part is weight decay part
	err2 = sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2));
	err2 = err2 * lambda / 2;
	% the third part of overall cost function is the sparsity part
	err3 = zeros(hiddenSize, 1);
	err3 = err3 + sparsityParam .* log(sparsityParam ./ pj) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - pj));
	cost = err + err2 + beta * sum(err3);
  
end
 
%-------------------------------------------------------------------
% This function calculate Sigmoid
%
function sigm = sigmoid(x)
	sigm = 1 ./ (1 + exp(-x));
end
 
%-------------------------------------------------------------------
% This function calculate dSigmoid
%
function dsigm = dsigmoid(a)
	dsigm = a .* (1.0 - a); 
end
 
%-------------------------------------------------------------------
% This function return the activation of each layer
%
function [ainput, ahidden, aoutput] = getActivation(W1, W2, b1, b2, input)
 	ainput = input;
	ahidden = bsxfun(@plus, W1 * ainput, b1);
	ahidden = sigmoid(ahidden);
	aoutput = bsxfun(@plus, W2 * ahidden, b2);
	aoutput = sigmoid(aoutput);
end

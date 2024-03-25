function [cost,grad] = back_propagation(theta, visibleSize, hiddenSize, ...
					    lambda, sparsityParam, beta, data)
	
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


	% Initialize them to zeros
	W1grad = zeros(size(W1)); 
	W2grad = zeros(size(W2));
	b1grad = zeros(size(b1)); 
	b2grad = zeros(size(b2));

	% Computing gradient

	[nFeatures, nSamples] = size(data);
	[a1, a2, a3] = getActivation(W1, W2, b1, b2, data);
        
	    delta3 = -(data - a3) .* dsigmoid(a3);
        pj = sum(a2, 2) ./ nSamples;
    	delta2 = bsxfun(@plus, (W2' * delta3), beta .* (-sparsityParam ./ pj + (1 - sparsityParam) ./ (1 - pj))); 
    	delta2 = delta2 .* dsigmoid(a2);
    	nablaW1 = delta2 * a1';
    	nablab1 = delta2;
    	nablaW2 = delta3 * a2';
    	nablab2 = delta3;
 
    	W1grad = nablaW1 ./ nSamples + lambda .* W1;
    	W2grad = nablaW2 ./ nSamples + lambda .* W2;
    	b1grad = sum(nablab1, 2) ./ nSamples;
    	b2grad = sum(nablab2, 2) ./ nSamples;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc). Specifically, we will unroll
% your gradient matrices into a vector.
 
    grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
    cost = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
					     lambda, sparsityParam, beta, data);

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

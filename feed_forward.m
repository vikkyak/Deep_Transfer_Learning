function [activation] = feed_forward(theta, hiddenSize, visibleSize, data)

	% theta: trained weights from the autoencoder
	% visibleSize: the number of input units  
	% hiddenSize: the number of hidden units  
	% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
	% We first convert theta to the (W1, W2, b1, b2) matrix/vector format

	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);


	W = [b1 W1];
    %size(W)
	data = [ones(1,size(data,2));data];
    %size(data)
	activation = sigmoid(W*data);

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

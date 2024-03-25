function [stack] = stackFormation(theta,networkConfig)
	
	L = length(networkConfig)-1;
	inputSize  = networkConfig(1);
	hiddenSizeL = [];     
	for i = 1:L
		hiddenSizeL(i) = networkConfig(i+1);
	end

	stack = cell(L,1);
	
	for i=1:L				
		if(i==1)
			stack{i}.w = reshape(theta(1:hiddenSizeL(i)*inputSize), ...
            					hiddenSizeL(i), inputSize);
        		stack{i}.b = theta(2*hiddenSizeL(i)*inputSize+1:2*hiddenSizeL(i)*inputSize+hiddenSizeL(i));
		else
			stack{i}.w = reshape(theta(1:hiddenSizeL(i)*hiddenSizeL(i-1)), ...
            					hiddenSizeL(i), hiddenSizeL(i-1));
        		stack{i}.b = theta(2*hiddenSizeL(i)*hiddenSizeL(i-1)+1:2*hiddenSizeL(i)*hiddenSizeL(i-1)+hiddenSizeL(i));
		end	
	end
end 

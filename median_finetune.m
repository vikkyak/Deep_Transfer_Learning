function [finetuned_stack, clust_param] = median_finetune(theta, networkConfig, data, group, max_iter, lambda)
	
	if nargin < 5
    		options.maxIter = 300; %%300;% Maximum number of iterations of L-BFGS to run
    		lambda = 1e-4;
	elseif nargin < 6
    		options.maxIter = max_iter;
    		lambda = 1e-4;    
	else
    		options.maxIter = max_iter;    
	end

	stack = stackFormation(theta,networkConfig);

	hlevel_data = stackedAE_Out(data, stack);
	[op_nodes num_samples] = size(hlevel_data);
	unique_group = unique(group);
	num_classes = length(unique_group);
	class_data = cell(1,num_classes);
	pred_op = zeros(op_nodes,num_samples); 
	class_size = zeros(1,num_classes);
	addpath minFunc/
	options.Method = 'lbfgs'; 
	clust_param = {};
	%options.display = 'on';
	%options.TolX = 1e-12;

	
    	for i = 1:num_classes
        	ind = find(group == unique_group(i));
        	class_data{i} = hlevel_data(:,ind);
        	class_size(i) = size(class_data{i},2);
      		%class_median{i} = median(class_data{i},2);
       		pred_op(:,ind) = repmat(median(class_data{i},2),1,class_size(i));
   	end
    
    	[stackparams, netconfig] = stack2params(stack);
    	%[cost, grad] = feedforwardnet_Cost(stack, lambda, data, desired_out);
    	[updated_params, cost] = minFunc( @(p) ...
       				feedforwardnet_Cost(p, netconfig, lambda, data, pred_op) , ...
        			stackparams, options);

    
	finetuned_stack = params2stack(updated_params, netconfig);
end





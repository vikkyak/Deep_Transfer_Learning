function [normData,min_of_all,max_of_all] = Zero_One_Normalization(data,max_of_all,min_of_all)

	if nargin == 1
		min_of_all = min(min(data));
		max_of_all = max(max(data));
	end

	normData = (data - min_of_all*(ones(size(data))))./(max_of_all - min_of_all);
end

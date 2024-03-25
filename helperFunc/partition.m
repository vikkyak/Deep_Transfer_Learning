function [labData,unlabData] = partition()
    
	X1 = load('Colon.mat');
	X2 = load('Uterus.mat');
	X3 = load('Breast.mat');
	X4 = load('Kidney.mat');
	X5 = load('Lung.mat');
	X6 = load('Omentum.mat');
	X7 = load('Prostate.mat');
	X8 = load('endometrium.mat');
	X9 = load('Omentum.mat');

	labData  = [X1;X2];
    	labels = [zeros(size(X1,1),1);ones(size(X2,1),1)];
    	labData = [labData labels];
	unlabData = [X3;X4;X5;X6;X7;X8;X9];

end

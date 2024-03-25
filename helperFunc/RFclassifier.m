function [auc_RF] = RFclassifier(traindata,trainlabels,testdata,testlabels)
 
	addpath('libsvm');     
     
  	B = TreeBagger(1000,traindata,trainlabels);
  	[Yfit,scores,stdevs] = predict(B,testdata);
     	Yfit=cell2mat(Yfit);
     	Yfit=str2num(Yfit);
      	acc_RF = mean(testlabels(:) == Yfit(:));
     	[~,~,T,auc_RF] = perfcurve(testlabels,scores(:,1),1);
        auc_RF = max(auc_RF);
    
end


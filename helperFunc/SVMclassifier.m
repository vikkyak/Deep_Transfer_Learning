function [auc_lin,auc_rbf] = SVMclassifier(traindata,trainlabels,testdata,testlabels) 

    addpath('h')    
    
    bestcv = 0; bestc = 0;
    %%%%%%%%%tuning of parameters%%%%%%%%%
    for log2c = 1:15
        cmd = ['-t 0 -v 5 -c ', num2str(2^log2c), ' -q'];
        cv = svmtrain(trainlabels, traindata, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = log2c;
        end
        fprintf('%g %g (best c = %g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cur_bestc = bestc;
    
    % Finer grid search for finding best parameters
    for i = 1:7
        log2c = cur_bestc - 1 + 0.25*i;
        cmd = ['-t 0 -v 5 -c ', num2str(2^log2c), ' -q'];
        cv = svmtrain(trainlabels, traindata, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = 2^log2c;
        end
        fprintf('%g %g (best c = %g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end
    
    cmd = ['-t 0 -c ', num2str(bestc),' -b 1'];
    
    % Final model made ready
    clsfr_model = svmtrain(trainlabels,traindata, cmd);
    
    [pred_label, acc1, prob_est1] = svmpredict(testlabels, testdata, clsfr_model,'-b 1'); %%%'-b 1'
    [~,~,T,auc_lin] = perfcurve(testlabels',prob_est1(:,1),2);
     
    
    
    
    %%
    bestcv = 0; bestc = 0;
    % with RBF kernel
    % Finding the best parameters through grid search
    for log2c = 1:15
        for log2g = -15:-1
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
            cv = svmtrain(trainlabels, traindata, cmd);
            if (cv >= bestcv)
                bestcv = cv; bestc = log2c; bestg = log2g;
            end
            fprintf('%g %g %g (best c = %g, best g = %g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    cur_bestc = bestc; cur_bestg = bestg;
    
    % Finer grid search for finding best parameters
    for i = 1:7
        log2c = cur_bestc - 1 + 0.25*i;
        for k = 1:7
            log2g = cur_bestg - 1 + 0.25*k;
            cmd = ['-t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -q'];
            cv = svmtrain(trainlabels, traindata, cmd);
            if (cv >= bestcv)
                bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
            end
            fprintf('%g %g %g (best c = %g, best g = %g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
        end
    end
    
    cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg),' -b 1'];
    
    % Final model made ready
    clsfr_model = svmtrain(trainlabels, traindata, cmd);
    
    [pred_label, acc2, prob_est2] = svmpredict(testlabels, testdata, clsfr_model,'-b 1');%%, '-b 1');
    
    [~,~,T,auc_rbf] = perfcurve(testlabels,prob_est2,2);
    
end


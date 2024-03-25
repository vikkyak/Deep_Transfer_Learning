function [predict,threshold]=findMIE2(classinfo,featureinfo)
[m,n] = size(featureinfo) ; 
[sortedarr,I]=sort(featureinfo);
x=size(featureinfo,1);
errordown=zeros(x+1,1);
errorup=zeros(x+1,1);

for z=0:x
    if z==0 
        errordown(1)=sum(1-classinfo(I(1:x)));
        errorup(1)=sum(classinfo(I(1:x)));
    elseif z==x
        errordown(x+1)=sum(classinfo(I(1:x)));
        errorup(x+1)=sum(1-classinfo(I(1:x)));
    else
        errordown(z+1)=sum(1-classinfo(I(z+1:x)))+sum(classinfo(I(1:z)));
        errorup(z+1)=sum(classinfo(I(z+1:x)))+sum(1-classinfo(I(1:z)));
    end
end

[minvaldown,indexdown]=min(errordown);   
[minvalup,indexup]=min(errorup);   

if (minvaldown<=minvalup)
    index=indexdown;
    predict(I(1:index-1))=1;
    predict(I(index:x))=0;
    threshold(1)=0;
    if index == 1
        threshold(2) = sortedarr(index) - (sortedarr(index+1)+sortedarr(index))/2;
    elseif index == m+1
        threshold(2)= sortedarr(index-1) + (sortedarr(index-2)+sortedarr(index-1))/2;
    else
        threshold(2)=(sortedarr(index-1)+sortedarr(index))/2;
    end
    
else 
    index=indexup;
    predict(I(1:index-1))=0;
    predict(I(index:x))=1;
    threshold(1)=1;
    if index == 1
        threshold(2) = sortedarr(index) - (sortedarr(index+1)+sortedarr(index))/2;
    elseif index == m+1
        threshold(2)= sortedarr(index-1) + (sortedarr(index-2)+sortedarr(index-1))/2;
    else
        threshold(2)=(sortedarr(index-1)+sortedarr(index))/2;
    end
%    threshold(2)=(sortedarr(index-1)+sortedarr(index))/2;
end
end

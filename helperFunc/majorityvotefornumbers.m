function [a,threshold]=majorityvotefornumbers(A,B) %A is featuredata, B is classinfo
% a returns an array that indicates '1' when the sample is misclassified. 
% threshold returns information for testing error

a=zeros(size(A,1),1);
threshold=zeros(size(A,2),2);
%errorarr=zeros(size(A,2),1);
for i=1:size(A,2)
  X=A(:,i);
  Y=B;
  [ldaClass,threshold(i,:)]=findMIE2(Y,X);
  for j=1:size(A,1)
      a(j)=a(j)+ (ldaClass(j)==B(j));
  end 
end

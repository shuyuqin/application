function [ error ] = class_error( w,X,y )
    n=size(X,1);
    col=size(X,2)+1;
    xte(:,2:col)=X;
    xte(:,1)=1;
    preds=sign(xte*w');
    error=sum(abs(y-preds)/2)/n;
end


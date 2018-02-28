function [ logerr,pererr ] = crossvalidation( Data,n,max_its,eta )

row=size(Data,1);
col=size(Data,2);
k=floor(row/n);
log=zeros(1,n);
per=zeros(1,n);
for i=1:n
    s=k*(i-1)+1;
    e=k*i;
    Dtrain=Data;
    Dtrain(s:e,:)=[];
    Dtest= Data(s:e,:);
    X=Dtrain(:,1:col-1);
    y=Dtrain(:,col);
    w_init=col;
    [wlog, e_in, its] = logistic_reg( X, y, w_init, max_its, eta );
    Xt=Dtest(:,1:col-1);
    yt=Dtest(:,col);
    log(i)  = class_error( wlog,Xt,yt );
    rowT=size(Dtrain,1);
    Dper=[ones(rowT,1),Dtrain];
    [ wper, iterations ] = perceptron_learn( Dper,max_its );
    per(i)=class_error( wper,Xt,yt );
    
end
   logerr=sum(log)/n;
   pererr=sum(per)/n;
end

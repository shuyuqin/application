function [ w, e_in, its ] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)

    w=zeros(1,w_init);
    its=0;
    bound=1e-6;
    N = size(X,1);
    X=[ones(N,1),X];
    while its <max_its
        temp_exp=exp(y.*(X*w'));
        vt=(y./(1+temp_exp))'*X/N;
        w=w+eta*vt;
        if ~sum(abs(vt)>bound)
            disp(its);
            break;
        end
        its =its+ 1;
        
    end
    e_in=mean(log(1+1./temp_exp));

    
    
end


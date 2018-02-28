function [ w iterations ] = perceptron_learn( data_in,max_its )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for

[num_sample,d]=size(data_in);
w=zeros(1,d-1);        
iterations=0;
while(sum(sign(data_in(:,1:d-1)*w')~=data_in(:,d)))&&(iterations<max_its)
    for i=1:num_sample
        if sign(data_in(i,1:d-1)*w')~=data_in(i,d)
            break
        end
    end
    w=w+data_in(i,1:d-1)*data_in(i,d);
    iterations=iterations+1;
end
end







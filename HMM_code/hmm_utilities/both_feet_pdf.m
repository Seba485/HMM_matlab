function [likelihood] = both_feet_pdf(x,param)
%pdf for the both feet class: [likelihood] = both_feet_pdf(x,param)
%function: A*exp(B.*(x-1)) + A_1*exp(B_1.*(x-1));
%input: x--> data (or column vector data)
%       param--> [A, B, A_1, B_1] where A is the amplitude of the
%       exponential and B is the speed of decay
%output: likelihood of the data

    A = param(1);
    B = param(2);
    A_1 = param(3);
    B_1 = param(4);

    likelihood = A*exp(B.*(x-1)) + A_1*exp(B_1.*(x-1));
end
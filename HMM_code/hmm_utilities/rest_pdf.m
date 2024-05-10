function [likelihood] = rest_pdf(x,param)
%pdf for the rest class: [likelihood] = rest_pdf(x,param)
%function: both_hand_pdf + both_feet_pdf;
%input: x--> data (or column vector data)
%       param--> [A, B, A_1, B_1] where A is the amplitude of the
%       exponential and B is the speed of decay
%output: likelihood of the data

    A = param(1);
    B = param(2);
    A_1 = param(3);
    B_1 = param(4);

    likelihood = A*exp(-B.*x) + A_1*exp(-B_1.*x) + A*exp(B.*(x-1)) + A_1*exp(B_1.*(x-1));
    likelihood = likelihood./2;
end
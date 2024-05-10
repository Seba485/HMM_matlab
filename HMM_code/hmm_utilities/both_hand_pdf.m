function [likelihood] = both_hand_pdf(x,param)
%pdf for the both hand class: [likelihood] = both_hand_pdf(x,param)
%function: A*exp(-B.*x) + A_1*exp(-B_1.*x);
%input: x--> data (or column vector data)
%       param--> [A, B, A_1, B_1] where A is the amplitude of the
%       exponential and B is the speed of decay
%output: likelihood of the data

    A = param(1);
    B = param(2);
    A_1 = param(3);
    B_1 = param(4);

    likelihood = A*exp(-B.*x) + A_1*exp(-B_1.*x);
end
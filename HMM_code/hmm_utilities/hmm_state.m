function [likelihood] = hmm_state(x,state,varargin)
%pdf for the states 'task_1' 'task_2' 'rest'
%[likelihood] = hmm_state(x,state,'param',param_array)
%function: sum of two exponential --> both hand: A*exp(-B.*x) + A_1*exp(-B_1.*x);
%                                     both feet: A*exp(B.*(x-1)) + A_1*exp(B_1.*(x-1));
%                                     rest: both feet + both hand / 2
%input: x--> data (or column vector of data)
%       state--> string that identify the class state
%       optional: param--> [A, B, A_1, B_1] where A is the amplitude of the
%       exponential and B is the speed of decay
%output: likelihood of the data
    
    custom_param = false;
    if nargin>3
        for k = 1:numel(varargin)
            if strcmpi(varargin{k}, 'param')
                if isvector(varargin{k+1}) & length(varargin{k+1})==4
                    A = varargin{k+1}(1); B = varargin{k+1}(2); 
                    A_1 = varargin{k+1}(3); B_1 = varargin{k+1}(4); 
                    custom_param = true;
                end
            end
        end
    end
    
    switch state
        case 'task_2'
            if custom_param == false
                A = 10; B = 20;
                A_1 = 5; B_1 = 8;
            end

            likelihood = A*exp(-B.*x) + A_1*exp(-B_1.*x);
        case 'task_1'
            if custom_param == false
                A = 10; B = 20;
                A_1 = 5; B_1 = 8;
            end

            likelihood = A*exp(B.*(x-1)) + A_1*exp(B_1.*(x-1));
        case 'rest'
            if custom_param == false
                A = 10; B = 40;
                A_1 = 5; B_1 = 8;
            end

            likelihood = A*exp(-B.*x) + A_1*exp(-B_1.*x) + A*exp(B.*(x-1)) + A_1*exp(B_1.*(x-1));
            likelihood = likelihood./2;
        otherwise
            warning('The selected state does not exist, try to check the variable state')
    end
end
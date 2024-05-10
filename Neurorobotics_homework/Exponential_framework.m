function [D] = Exponential_framework(D_init,pp,alpha)
%[D] = Exponential_framework(pp,alpha)
%Applly the exponential framework to the posterior probability
%D_init --> precedent value of the framework
%pp --> posterior probability (output of the classifier)
%alpha --> past value weigth of the smoother
%OUTPUT: D --> framework output related to the first class for that pp
    


    D = D_init*alpha + pp*(1-alpha);
            
end




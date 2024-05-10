function [y] = Dynamic_framework(y_init,pp,parameters)
%[y] = Dynamic_framework(pp,parameters)
%Applly the dynamic framework to the posterior probability
% REFERENCE: Tonin L et al. The role of the control framework for continuous tele-operation of a BMI driven mobile robot.
%            IEEE Transactions on Robotics, 36(1):78-91, 2020. doi: 10.1109/TRO.2019.2943072
%y_init --> previous output of the framework
%pp --> posterior probability point (output of the classifier)
%parameters --> [y_stable w psi chi phi] parameter of the framework; 
%CODE --> struct with the experimental codes
%OUTPUT: y --> framework output for that pp
    
    y_limit = [0 1];
    y_stable = parameters(1);
    w = parameters(2);
    psi = parameters(3);
    chi = parameters(4);
    phi = parameters(5);


            
    % Free Force
    if y_init<y_stable-w && y_init>=y_limit(1)
        F_free = -sin(y_init*pi/(y_stable-w));
    end
    if y_init>=y_stable-w && y_init<=y_stable+w
        F_free = -psi*sin(pi*(y_init-y_stable)/w);
    end
    if y_init>y_stable+w && y_init<=y_limit(2)
        F_free = sin((y_init-y_stable-w)*pi/(y_stable-w));
    end
    
    % BMI Frorce
    F_bmi = 6.4*(pp-0.5)^3 + 0.4*(pp-0.5); %cubic transformation
    
    % update of the framework
    dy = chi*(phi*F_free + (1-phi)*F_bmi);

    y = y_init+dy;

    if y>y_limit(2)
        y = y_limit(2);
    end
    if y<y_limit(1)
        y = y_limit(1);
    end
        
end
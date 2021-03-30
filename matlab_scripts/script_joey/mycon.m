function [c,ceq] = mycon(x,w,z,k,eta,self_con)

ceq = self_con - calculate_self_cons ( x , w , z , k , eta );

c = [];

% ceq = sum(w - max(0,(k(3).*z.^2 + k(2).*z + k(1)).*x)) - self_con*max(0,(k(3).*z.^2 + k(2).*z + k(1)).*x);
% 
% 
% c = -(k(3).*z.^2 + k(2).*z + k(1)).*x;


end

function [ y ] = calculate_self_cons ( x , w , z , k, eta )
% CALCULATE_SELF_CONS evaluates the self-consumption of power system with
% PV and load
% y = calculate_self_cons ( x , w , z , k, eta )
% 
% OUTPUT
%   y - is the self consumption
% INPUT
%   x - is the nominal power of the PV plant
%   w - is the load profile
%   z - is the irradiance
%   k - is the 3-by-1 vector of parameters for generating power from
%       irradiance
% eta - is the efficiency of the PV plant

P_pv = max(0,eta*(k(3).*z.^2 + k(2).*z + k(1)).*x);

a = sum(P_pv);

b = P_pv;

b((b - w) > 0) = w ( (b - w) > 0 );

y = sum(b)/a;

end
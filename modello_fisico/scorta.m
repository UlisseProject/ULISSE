i=iPV-i0*(exp(V/(n*Vt))-1);

iSC=iSC_ref*Gtot/Gref*(1+alphaISC*(Tc-Tc_ref));
VOC=VOC_ref*(1+betaVOC*(Tc-TC_ref))+A*log(Gtot/Gref);

Tc_ref=25;
Gref=1000;
iPV_ref=;
alphaISC=;
iPV=Gtot/Gref*(iPV_ref*(1+alphaISC*(Tc-Tref)));
i0_ref=;
Tref=;
Eg_ref=;
Eg=1.17-4.73*10^(-4)*Tc^2/(Tc+636);
n=;
k=8.6173324; %Boltzmann constant
i0=io_ref*(Tc/Tref)^3*exp(Eg_ref/(n*k*Tref)-Eg/(n*k*Tc));


n = 1.0134; % ideality factor (diode factor)



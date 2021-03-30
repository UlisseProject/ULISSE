cicli=[0 23 50 78 100 148 175 200 275 300];
Ah=[9.82 9.72 9.65 9.6 9.58 9.48 9.42 9.4 9.22 9.18];
SOC=Ah/max(Ah);
coeff_cicli=polyfit (cicli, SOC, 2);
x=linspace(0,350, 350);
y=polyval(coeff_cicli, x);
plot(x, y, cicli, SOC)

eff_cicli=polyval(coeff_cicli, n_cicli);
%nota che questa va solo moltiplicata alla carica, mentre l'altra
%efficienza va conteggiata alla scarica, come mi hi spiegato tut stesso

SOC_previous=SOC_input;
n_cicli;%input
%elabori tutto e sputi fuori la nuova SOC e quindi abbiamo
if P_input>0
n_cicli=n_cicli+(SOC-SOC_previous)/100;
end



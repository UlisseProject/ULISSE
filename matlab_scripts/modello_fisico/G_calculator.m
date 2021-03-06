function [G, theta, dHr] = G_calculator(Location, anno, m_inizio, m_fine, d_inizio, d_fine, beta, gamma, Gdiff, Gdni, Grefl)


L = 2.5e-3; %valore di default in metri dello spessore del vetro
K = 10; %valore in metri del fattore di estinzione del vetro
n_air = 1.0002926; 
n_glass = 1.58992; % da paper

Albedo =0.1;
z1 =zeros(24*60, 1);
alpha1 =zeros(24*60, 1);
GHI1 =zeros(24*60, 1);
DNI1 =zeros(24*60, 1);
DHI1 =zeros(24*60, 1);
GR1 =zeros(24*60, 1);

z = [];
alpha = [];
GHI = []; 
DNI = [];
DHI = [];
GR = [];

giorni_mese=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
i=1;

if nargin == 8
    for m= m_inizio:m_fine
        if (i~=1)
            d_inizio = 1; 
        end
        for d = d_inizio:giorni_mese(m)

       
            DN = datenum(anno, m,d):1/(24*60):datenum(anno, m, d, 23, 59, 59);
            Time = pvl_maketimestruct(DN, 1);
            [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location);
            [SunAz1, SunEl1, ApparentSunEl1]=pvl_spa(Time, Location);
            [ClearSkyGHI, ClearSkyDNI, ClearSkyDHI]= pvl_clearsky_ineichen(Time, Location);
  
            
%             z(:, i)=deg2rad(90- SunEl1);
%             alpha(:, i)=deg2rad(SunAz1-180);
%             GHI(:, i)=ClearSkyGHI;
%             DNI(:, i)=ClearSkyDNI;
%             DHI(:, i)=ClearSkyDHI;
            z1 = deg2rad(90- SunEl1);
            alpha1 = deg2rad(SunAz1-180);
            GHI1 = ClearSkyGHI;
            DNI1 =ClearSkyDNI;
            DHI1 =ClearSkyDHI;
            GR1(:, i) = pvl_grounddiffuse(rad2deg(beta), GHI1, Albedo);

            GHI1(isnan(GHI1))=0;
            DNI1(isnan(DNI1))=0;
            DHI1(isnan(DHI1))=0;
            GR1(isnan(GR1))=0;
            
            z = [z, z1]; 
            alpha = [alpha, alpha1]; 
            DNI = [DNI, DNI1]; 
            DHI = [DHI, DHI1]; 
            GR = [GR, GR1]; 
            
            
            i=i+1;   
            
            if (d == d_fine)&&(m == m_fine)
                break
            end
        end
    end

    Gdiff = DHI;
    Gdni = DNI; 
    Grefl = GR; 

elseif nargin == 11
    
    for m= m_inizio:m_fine
        
        if (i~=1)
            d_inizio = 1; 
        end
        for d = d_inizio:giorni_mese(m)

       
            DN = datenum(anno, m,d):1/(24*60):datenum(anno, m, d, 23, 59, 59);
            Time = pvl_maketimestruct(DN, 1);
            [SunAz, SunEl, ApparentSunEl, SolarTime]=pvl_ephemeris(Time, Location);
            [SunAz1, SunEl1, ApparentSunEl1]=pvl_spa(Time, Location);
            % z1 is the zenith
            z1 =deg2rad(90- SunEl1);
            alpha1=deg2rad(SunAz1-180);
            
            z = [z, z1]; 
            alpha = [alpha, alpha1]; 
            

            i=i+1;
            
            if (d == d_fine)&&(m == m_fine)
                break
            end
        end
    end

else
    disp(" !!! ERRORE: argomenti non sufficienti !!!"); 
    G = 0; 
    theta = 0; 
    dHr = 0; 
    return; 
end

% ERROR : SunEl1 is only 
% the last one of the loop above
alt_sol=deg2rad(SunEl1);
theta=acos(cos(alt_sol).*cos(alpha-gamma).*sin(beta)+sin(alt_sol).*cos(beta));


theta_r = asin(n_air/n_glass.*sin(theta));
 
Gtot = Gdni.*cos(theta) + Gdiff.*(1+cos(beta))/2 + Grefl*(1-cos(beta))/2; 

tau= @(theta, theta_r) exp((-K*L./cos(theta_r))).*(1-1/2.*(((sin(theta_r-theta)).^2./(sin(theta_r+theta)).^2+(tan(theta_r-theta)).^2./(tan(theta_r+theta)).^2)));

tau_b=tau(theta, theta_r);
beta_deg=rad2deg(beta);

theta_equiv_diff_deg= 59.7 - 0.1388*beta_deg + 0.001497*beta_deg^2;
%theta_equiv_diff_deg= 90 - 0.5788*beta_deg + 0.002693*beta_deg^2;

theta_equiv_diff=deg2rad(theta_equiv_diff_deg);
theta_r_equiv = asin(n_air/n_glass.*sin(theta_equiv_diff));

tau_d=tau(theta_equiv_diff, theta_r_equiv);
% tau_0=exp((-K*L)).*(1-((n_glass - 1)./(n_glass + 1)).^2);
% 
% K_tau_b=tau_b/tau_0;
% K_tau_d=tau_d/tau_0;

% tau_d = 1; 
G = tau_b.*Gdni.*cos(theta) + tau_d.*Gdiff.*(1+cos(beta))/2; 

dHr = Time.hour+Time.minute./60+Time.second./3600;






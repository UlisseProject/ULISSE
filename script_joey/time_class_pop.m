function [fasce_h,fasce_q] = time_class_pop(year,sd)
%TIME_CLASS_POP produces a file and 2 variables in workspace with time 
% peak/off-peak classification 
% INPUT
%       year:   
%       sd:     Special Days: special holidays (char array)
%               1 Jan., 6 Jan., 25 Apr., 1 May, 2 Jun., 15 Aug., 01 Nov., 
%               8 Dec., 25 Dec., 26 Dec., and Easter Monday.
%               
% OUTPUT
%       output is time_class_pop_'year'.txt
%       and 2 variables fasce_h and fasce_q in workspace
%       1 stands for P (peak)
%       2 stands for OP (off-peak)
% 
%       Time class.
%  ========================
%         Mon - Fri
% OP:   fr 00.00   to 08.00
%  P:   fr 08.00   to 20.00
% OP:   fr 20.00   to 24.00
%
%         Sat - Sun
% OP:   fr 00.00   to 24.00
% 
% NO HOLIDAYS
% =========================
% 
% Example:
% 
% year = 2019;
% sd = [];
% [fasce_h,fasce_q] = time_class_pop(year,sd);

%%

if ~isempty(sd)
    [y,sdm,sdd] = datevec(sd,'dd/mm/yyyy');
    clear y;
else
    sdm = [];
    sdd = [];
end

%%
% count = 0;
year_d = [];
year_h = [];
year_dh = [];
year_q = [];
fasce_h = [];
fasce_q = [];
%%
for h=1:12, % cycle on months
    cal2 = calendar(year,h);
    day = sort(cal2(find(cal2 ~= 0)));
    strday = datestr(datenum(year,h,day(:)),24);

    % Creates daily time classifications for current month
    % Criterion:
    % Mon - Fri
    % OP:   fr 00.00   to 08.00
    %  P:   fr 08.00   to 20.00
    % OP:   fr 20.00   to 24.00
    %
    %
    % Sat - Sun
    % OP:   fr 00.00   to 24.00

    fasce = [];
    calwd = cal2(:,2:6);
    wddays = sort(calwd(find(calwd ~= 0)));
    for i = 1:size(wddays,1),
        % P:   da 08.00    a 20.00
        fasce(((wddays(i)-1)*24+1)+8:((wddays(i)-1)*24+1)+19,1) = 1;
        % OP:   da 20.00    a 24.00
        fasce(((wddays(i)-1)*24+1)+20:((wddays(i)-1)*24+1)+23,1) = 2;
        % OP:   da 00.00    a 08.00
        fasce(((wddays(i)-1)*24+1):((wddays(i)-1)*24+1)+7,1) = 2;
    end
    calsats = cal2(:,end);
    sats = sort(calsats(find(calsats ~= 0)));
    for i = 1:size(sats,1),
        % OP:   da 00.00    a 24.00
        fasce(((sats(i)-1)*24+1):((sats(i)-1)*24+1)+23,1) = 2;
    end
    calsuns = cal2(:,1);
    suns = sort(calsuns(find(calsuns ~= 0)));
    for i = 1:size(suns,1),
        % OP:   da 00.00    a 24.00
        fasce(((suns(i)-1)*24+1):((suns(i)-1)*24+1)+23,1) = 2;
    end

    % Modifica fasce per giorni festivi straordinari nel mese corrente
    if ~isempty(find(sdm == h))
        idx = find(sdm == h);
        for i = 1:length(idx),
            fasce(((sdd(idx(i))-1)*24+1):((sdd(idx(i))-1)*24+1)+23,1) = 2;
        end
    end

    for k=1:size(day,1),
        for t=1:24,
            date((k-1)*96+(t-1)*4+1,:) = strday(k,:);
            quarter((k-1)*96+(t-1)*4+1,:) = datestr(datenum(year,h,k,t-1,15,0),'HH:MM');
            % f((k-1)*96+(t-1)*4+1,1) = fasce(count*24 + (k-1)*24 + t);
            f((k-1)*96+(t-1)*4+1,1) = fasce((k-1)*24 + t);
            date((k-1)*96+(t-1)*4+2,:) = strday(k,:);
            quarter((k-1)*96+(t-1)*4+2,:) = datestr(datenum(year,h,k,t-1,30,0),'HH:MM');
            % f((k-1)*96+(t-1)*4+2,1) = fasce(count*24 + (k-1)*24 + t);
            f((k-1)*96+(t-1)*4+2,1) = fasce((k-1)*24 + t);
            date((k-1)*96+(t-1)*4+3,:) = strday(k,:);
            quarter((k-1)*96+(t-1)*4+3,:) = datestr(datenum(year,h,k,t-1,45,0),'HH:MM');
            % f((k-1)*96+(t-1)*4+3,1) = fasce(count*24 + (k-1)*24 + t);
            f((k-1)*96+(t-1)*4+3,1) = fasce((k-1)*24 + t);
            date((k-1)*96+(t-1)*4+4,:) = strday(k,:);
            quarter((k-1)*96+(t-1)*4+4,:) = datestr(datenum(year,h,k,t,0,0),'HH:MM');
            %f((k-1)*96+(t-1)*4+4,1) = fasce(count*24 + (k-1)*24 + t);
            f((k-1)*96+(t-1)*4+4,1) = fasce((k-1)*24 + t);
            
            % orario
             hour((k-1)*24+t,:) = datestr(datenum(year,h,k,t,0,0),'HH:MM');
             date_h((k-1)*24+t,:) = strday(k,:);
        end
    end

    % count = count + day(end);
    year_d = [year_d;date];
    year_q = [year_q;quarter];
    year_h = [year_h;hour];
    year_dh = [year_dh;date_h];
    fasce_h = [fasce_h;fasce];
    fasce_q = [fasce_q;f];
    clear('date','date_h','hour','f');
    
end

save(strcat('fasce_pop_',num2str(year)),'fasce_h','fasce_q');

corrige = [1:1:(datenum(year,12,31,0,0,0)-datenum(year-1,12,31,0,0,0))]';
year_q(96*corrige(1:end),:) = repmat('24:00',size(corrige,1),1);

%%
% quarte-of-an-hour basis
fid = fopen(['fasce_pop_q_',num2str(year),'.txt'],'w+');
for r=1:size(year_d,1),
    fprintf(fid,'%s\t%s\t%.0f\n',year_d(r,:),year_q(r,:),fasce_q(r));
    
end
fclose(fid);

% hourly basis
corrige = [1:1:(datenum(year,12,31,0,0,0)-datenum(year-1,12,31,0,0,0))]';
year_h(24*corrige(1:end),:) = repmat('24:00',size(corrige,1),1);


fid = fopen(['fasce_pop_h_',num2str(year),'.txt'],'w+');

for r=1:size(year_dh,1),
    fprintf(fid,'%s\t%s\t%.1f\n',year_dh(r,:),year_h(r,:),fasce_h(r));
end
fclose(fid);
[data,txt] = xlsread('C:\Users\yjiaoneal\Desktop\工作\资产配置\1111\profit_weekly.csv');

txt = txt;
[a,b] = size(txt);
txt = txt(2:a,1:1)

[a,b] = size(data);
for i=1:12:(a-104)
    
     d = data( i:( i+104 ), 1:3 );
     date = txt(i+104);
     
     txt(i + 104);
     txt(i);
     
     d_cov = cov(d);
     d_mean = mean(d);
     
     [risk, returns ,weights] = frontcon(d_mean, d_cov, 100);
     
     
     for m = 1:10
         max = 0;
         min = 0;
         if m == 1
             max = 0.1;
             min = 0;
         end
         if m == 2
             max =  0.15;
             min = 0.1;
         end
         if m == 3
             max = 0.2;
             min = 0.15;
         end
         if m == 4
             max = 0.25;
             min = 0.2;
         end
         if m == 5
             max = 0.3;
             min = 0.25;
         end
         if m == 6
             max = 0.4;
             min = 0.3;
         end
         if m==7
             max = 0.5;
             min = 0.4;
         end
         if m == 8
             max = 0.6;
             min = 0.5;
         end
         if m == 9
             max = 0.7;
             min = 0.6;
         end
         if m == 10
             max = 0.8;
             min = 0.7;
         end
         
        sh = 0;
        final_risk = 0;
        final_return = 0;
        final_ws = [];
        for j = 1:100
            r = risk(j);
            re = returns(j);
            ws = weights(j,:);
         
            if ws(1) > 0.2
                continue
            end
         
            if ws(1) + ws(2) > max
                continue
            end
         
            if ws(1) + ws(2) < min
                continue
            end
         
            sharp = (re - 0.02/52) / r;
            if sharp > sh
                sh = sharp;
                final_risk = r;
                final_return = re;
                final_ws = weights(j,:);
            end
         
        end
     
     
         result = [];
         result = [result,  sh];
        result = [result , final_risk];
        result = [result, final_return];
        result = [result, final_ws];
     
        result
        date = strrep(date,'/','-');

        xlswrite(char(date),result, char(num2str(m)));
     end
end

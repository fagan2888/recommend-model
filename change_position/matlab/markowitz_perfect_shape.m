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
     
     sh = 0
     final_risk= []
     final_return = []
     final_ws = []
     
        for j = 1:100
            r = risk(j);
            re = returns(j);
            ws = weights(j,:);
         
            sharp = (re - 0.02/52) / r;
            if sharp > sh
                sh = sharp;
                final_risk = r;
                final_return = re;
                final_ws = weights(j,:);
            end
         
        end
    
       
     final_ws

end

th = 0:0.01:2*pi
r = exp(cos(th))-2*cos(4*th)+(sin(th/12)).^5;
polarplot(th, r)
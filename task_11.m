a = 1/2;
b = 1/3;
n = 17;
m = 7;
t = -5:0.01:5;
x = cos(t)+a*cos(m*t)+b*sin(n*t);
y = sin(t)+a*sin(m*t)+b*cos(n*t);
plot(x,y)
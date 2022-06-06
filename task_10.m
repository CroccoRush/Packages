a = input("Enter value of a:");
b = input("Enter value of b:");
phi_0 = input("Enter value of phi_0:");
n = input("Enter value of n:");
m = input("Enter value of m:");
phi = 0:0.05:2*pi;
x = a * sin(n * phi + phi_0);
y = b * sin(m * phi);
plot(x,y)
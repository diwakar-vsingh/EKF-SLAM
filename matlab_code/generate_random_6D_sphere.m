nPointsRand = 1000;
X = rand( 1, nPointsRand )-0.5;
Y = rand( 1, nPointsRand )-0.5;
Z = rand( 1, nPointsRand )-0.5;
theta = rand( 1, nPointsRand )-0.5;
phi = rand( 1, nPointsRand )-0.5;
lambda = rand( 1, nPointsRand )-0.5;
for i = 1:nPointsRand
    a = [X(i) Y(i) Z(i) theta(i) phi(i) lambda(i)];
    a = a/norm(a)*sqrt(12.59158724374398);
    X(i) = a(1); Y(i) = a(2); Z(i) = a(3);
    theta(i) = a(4); phi(i) = a(5); lambda(i) = a(6);
end
randSphere6D = [ X; Y; Z; theta; phi; lambda ];
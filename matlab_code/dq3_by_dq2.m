function dq3_by_dq2RES=dq3_by_dq2(q1_in)

q1.R=q1_in(1);
q1.X=q1_in(2);
q1.Y=q1_in(3);
q1.Z=q1_in(4);

dq3_by_dq2RES = ...
   [q1.R, -q1.X, -q1.Y, -q1.Z,
    q1.X,  q1.R,  q1.Z, -q1.Y,
    q1.Y, -q1.Z,  q1.R,  q1.X,
    q1.Z,  q1.Y, -q1.X,  q1.R];

return

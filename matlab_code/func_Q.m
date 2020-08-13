function [Q,G] = func_Q(Xv, u,Pn,delta_t, type)

if strcmp(type,'constant_position_and_orientation_location_noise')
    
    omegaOld=Xv(11:13);
    qOld=Xv(4:7);

    G=sparse(zeros(13,6));

    G(1:3,1:3)=eye(3)*delta_t;
    G(4:7,4:6)=dq_by_deuler(tr2rpy(q2tr(qOld)));
    
else

    omegaOld=Xv(11:13);
    qOld=Xv(4:7);
    qwt=v2q(omegaOld*delta_t);

    G=sparse(zeros(13,6));

    G(8:10,1:3)=eye(3);
    G(11:13,4:6)=eye(3);
    G(1:3,1:3)=eye(3)*delta_t;
    G(4:7,4:6)=dq3_by_dq1(qOld)*dqomegadt_by_domega(omegaOld,delta_t);

end

Q=G*Pn*G';
%% Team 19 Controls Part 1

%% Load Data

load ('TestTrack.mat');

bl = TestTrack.bl;       % Left Boundaries
br = TestTrack.br;       % Right Boundaries
cline = TestTrack.cline; % Center Line
theta = TestTrack.theta; % Center Line's Orientation

dt = 0.01;

%U = vector of constant steering angle, acceleration for given time

U = [repmat([-0.04, 2400],100,1);
    repmat([0, 2400],200,1);
    repmat([0.0, 2400],30,1);
    repmat([0.0, 2400],100,1);
    repmat([0.0, 2400],180,1);
    repmat([-0.0, 1800],130,1);
    repmat([-0.03,0],350,1);
    repmat([+0.05, 0],100,1);
    repmat([0.025, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0, 0],100,1);
    repmat([0.03, 0],100,1);
    repmat([0.03, 0],100,1);
    repmat([0.01, 0],100,1);
    repmat([0, 0],100,1);
    repmat([-0.02, 0],100,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.06 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.6 , 0],50,1);
    repmat([-0.06 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.04 , 0],50,1);
    repmat([-0.05 , 0],50,1);
    repmat([-0.05 , 0],50,1);
    repmat([-0.03 , 0],50,1);
    repmat([-0.01 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.03 , 0],50,1);
    repmat([0.05 , 0],50,1);
    repmat([0.07 , 0],50,1);
    repmat([0.09 , 0],50,1);
    repmat([0.07 , 0],50,1);
    repmat([0.05 , 0],50,1);
    repmat([0.03 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.0 , 0],100,1);
    repmat([0.01 , 0],100,1);
    repmat([0.01 , 0],100,1);
    repmat([0.02 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.02 , 0],100,1);
    repmat([0.01 , 0],200,1);
    repmat([0 , 5000],860,1)]; 
    
%time = size(U,1) * 0.01
    
x0 = [287 , 5 , -176 , 0 , 2 , 0];

[Y,T] = forwardIntegrateControlInput(U,x0);


Y;
U_in = U;
trajec = Y';
theta_traject = trajec(5,:);
trajec = trajec([1,3],:);

figure(1)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(trajec(1,:),trajec(2,:),'b--')

%% Initialize Constants

m    = 1400;                % Mass of Car
N_w  = 2.00;                % ?? 
f    = 0.01;                % ??
I_z  = 2667;                % Momemnt of Inertia
a    = 1.35;                % Front Axle to COM
b    = 1.45;                % Rear Axle to Com
B_y  = 0.27;                % Empirically Fit Coefficient
C_y  = 1.2;                % Empirically Fit Coefficient
D_y  = 0.70;                % Empirically Fit Coefficient
E_y  = -1.6;                % Empirically Fit Coefficient
S_hy = 0.00;                % Horizontal Offset in y
S_vy = 0.00;                % Vertical Offset in y
g    = 9.806;               % Graviational Constant

%% Initialize Time and Prediction Data

dt   = 0.01;                % Time Step

interp_size = 10;

bl = [interp(bl(1,:),interp_size);interp(bl(2,:),interp_size)];
br = [interp(br(1,:),interp_size);interp(br(2,:),interp_size)];
cline = [interp(cline(1,:),interp_size);interp(cline(2,:),interp_size)];
trajec = [interp(trajec(1,:),interp_size);interp(trajec(2,:),interp_size)];
theta_traject = interp(theta_traject,interp_size);
theta = interp(theta,interp_size);
cline = [interp(cline(1,:),interp_size);interp(cline(2,:),interp_size)];

figure(2)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(trajec(1,:),trajec(2,:),'b--')

nsteps = size(bl,2);
T = 0.0:dt:nsteps*dt;

nstates = 6;
ninputs = 2;


%% Initial Conditions

x0   =   287;
u0   =   5.0;
y0   =  -176;
v0   =   0.0;
psi0 =   2.0;
r0   =   0.0;

z0 = [x0, u0, y0, v0, psi0, r0];
           
%% PATH SEGMENT SELECTION
%Update to select new path (based on critical path indicies) within a for loop

i = 1;
nsteps = 50;
bl = bl(:,i:i+nsteps-1);
br = br(:,i:i:i+nsteps-1);
cline = cline(:,i:i+nsteps-1);
theta = theta(:,i:i+nsteps-1) ;
trajec = trajec(:,i:i+nsteps-1);
                      
%% Path Generation using NL Optimization

lowbounds = min(bl,br);
highbounds = max(bl,br);

[lb,ub]=bound_cons(nsteps,theta ,lowbounds, highbounds);


options = optimoptions('fmincon','SpecifyConstraintGradient',true,...
                       'SpecifyObjectiveGradient',true) ;
                   
endpoint = [trajec(1,nsteps),trajec(2,nsteps)];
xrefs = x0:(endpoint(1)-x0)/(nsteps-1):endpoint(1);
yrefs = y0:(endpoint(2)-y0)/(nsteps-1):endpoint(2);

states0 = [xrefs;u0*ones(1,nsteps);yrefs;v0*ones(1,nsteps);theta;r0*ones(1,nsteps)];
states0 = reshape(states0,1,nsteps*nstates);

X0 = [states0, repmat([-0.04,4000],1,nsteps-1)];

% cf=@costfun;
cf= @(z)costfun_segmt(z, trajec,nsteps);
nc=@(z)nonlcon(z,z0);

z=fmincon(cf,X0,[],[],[],[],lb',ub',nc,options);

Y=reshape(z(1:6*nsteps),6,nsteps)';
U=reshape(z(6*nsteps+1:end),2,nsteps-1)';

figure(3)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(Y(:,1),Y(:,3),'b')
plot(highbounds(1,:) , highbounds(2,:) , 'r')
plot(lowbounds(1,:) , lowbounds(2,:) , 'r')

%% REF PATH FOLLOWING WITHOUT OBSTACLES

%% Functions

function [lb,ub]=bound_cons(nsteps,theta ,lowbounds, highbounds) %,input_range

ub = [];
lb = [];

for i = 1:nsteps
    
ub = [ub,[highbounds(1,i), +inf,highbounds(2,i), +inf, theta(i)+pi/3, +inf]];

lb = [lb,[lowbounds(1,i), -inf, lowbounds(2,i), -inf, theta(i)-pi/3, -inf]];

end

ub = [ub,repmat([2500,0.5],1,nsteps-1) ]';
lb = [lb,repmat([-5000,-0.5],1,nsteps-1) ]';

end

function [J, dJ] = costfun_segmt(z,trajec,nsteps) %ADD X IN Y IN THETA IN
    if size(z,2) > size(z,1)
        z = z' ;
    end
    nsteps = (size(z,1)+2)/8 ;

    zx = z(1:6*nsteps) ;
    zu = z(6*nsteps+1:end) ;

    R = zeros(2*nsteps-2);

    nom=zeros(6*nsteps,1) ;
    nom(1:6:6*nsteps) =  trajec(1,nsteps) ; %REPLACE W X IDX 
    nom(3:6:6*nsteps+2) = trajec(2,nsteps) ;
    %nom(5:6:6*nsteps+4) = 1.8900 ;

    Q = diag(repmat([1/10,0,1/10,0,0,0],1,nsteps));  %OPTIMIZE TO MAXIMIZE SPEED

    J = zu'*R*zu+(zx-nom)'*Q*(zx-nom) ;
    dJ = [2*Q*zx-2*Q*nom;...
          2*R*zu]' ;
end

function [g,h,dg,dh]=nonlcon(z , x0)

    if size(z,2) > size(z,1)
        z = z' ;
    end
    
    nsteps = (size(z,1)+2)/8 ;
    dt = 0.01 ;

    zx = z(1:nsteps*6) ;
    zu = z(nsteps*6+1:end) ;
    
    g = [];
    dg = [];
    
    h = zeros(6*nsteps,1) ;
    dh = zeros(6*nsteps,8*nsteps-2);

    for i = 1:nsteps
        
        if i == 1
            h(1:6) = z(1:6,:) - x0' ;
            dh(1:6,1:6) = eye(6) ; 
        else
            h(6*i-5:6*i) = zx(6*i-5:6*i)-zx(6*i-11:6*i-6)-dt*odefun(zx(6*i-11:6*i-6),zu(2*i-3:2*i-2)) ;
                           
            dh(6*i-5:6*i,6*i-11:6*i) = [-eye(6)-dt*statepart_hand(zx(6*i-11:6*i-6),zu(2*i-3:2*i-2)),eye(6)] ;
            
            dh(6*i-5:6*i,6*nsteps+2*i-3:6*nsteps+2*i-2) = -dt*inputpart_hand(zx(6*i-11:6*i-6),zu(2*i-3:2*i-2)) ;
        end
        
    end

    dg = dg' ;
    dh= dh' ;
end

function dzdt = odefun(x,U)
%constants
Nw=2;
f=0.01;
Iz=2667;
a=1.35;
b=1.45;
By=0.27;
Cy=1.2;
Dy=0.7;
Ey=-1.6;
Shy=0;
Svy=0;
m=1400;
g=9.806;


%generate input functions
delta_f=U(1); %interp1(T,U(:,1),t,'previous','extrap');
F_x=U(2); %interp1(T,U(:,2),t,'previous','extrap');

%slip angle functions in degrees
a_f=rad2deg(delta_f-atan2(x(4)+a*x(6),x(2)));
a_r=rad2deg(-atan2((x(4)-b*x(6)),x(2)));

%Nonlinear Tire Dynamics
phi_yf=(1-Ey)*(a_f+Shy)+(Ey/By)*atan(By*(a_f+Shy));
phi_yr=(1-Ey)*(a_r+Shy)+(Ey/By)*atan(By*(a_r+Shy));

F_zf=b/(a+b)*m*g;
F_yf=F_zf*Dy*sin(Cy*atan(By*phi_yf))+Svy;

F_zr=a/(a+b)*m*g;
F_yr=F_zr*Dy*sin(Cy*atan(By*phi_yr))+Svy;

F_total=sqrt((Nw*F_x)^2+(F_yr^2));
F_max=0.7*m*g;

if F_total>F_max
    
    F_x=F_max/F_total*F_x;
    F_yr=F_max/F_total*F_yr;
    
end

%vehicle dynamics
dzdt =   [x(2)*cos(x(5))-x(4)*sin(x(5));...
          (-f*m*g+Nw*F_x-F_yf*sin(delta_f))/m+x(4)*x(6);...
          x(2)*sin(x(5))+x(4)*cos(x(5));...
          (F_yf*cos(delta_f)+F_yr)/m-x(2)*x(6);...
          x(6);...
          (F_yf*a*cos(delta_f)-F_yr*b)/Iz];
end

function [dx] = odefunOLD(Y,Uin)

% m    = 1400;                % Mass of Car
% N_w  = 2.00;                % ?? 
% f    = 0.01;                % ??
% I_z  = 2667;                % Momemnt of Inertia
% a    = 1.35;                % Front Axle to COM
% b    = 1.45;                % Rear Axle to Com
% B_y  = 0.27;                % Empirically Fit Coefficient
% C_y  = 1.35;  1.2               % Empirically Fit Coefficient
% D_y  = 0.70;                % Empirically Fit Coefficient
% E_y  = -1.6;                % Empirically Fit Coefficient
% g    = 9.806;               % Graviational Constant
% 
% alpha_f = (Uin(2) - atan((Y(4)+a*Y(6))/Y(2)));
% alpha_r = (- atan((Y(4)-b*Y(6))/Y(2)));
% 
% psi_yf = ((1-E_y)*alpha_f + E_y/B_y*atan(B_y*alpha_f));  % S_hy = 0
% psi_yr = ((1-E_y)*alpha_r + E_y/B_y*atan(B_y*alpha_r));  % S_hy = 0
% 
% 
% F_yf = (b/(a+b)*m*g*D_y*sin(C_y*atan(B_y*psi_yf))); %S_vy = 0;
% F_yr = (a/(a+b)*m*g*D_y*sin(C_y*atan(B_y*psi_yr))); %S_vy = 0;
% 
% dx = [          Y(2)*cos(Y(5))-Y(4)*sin(Y(5));
%            1/m*(-f*m*g+N_w*Uin(1)-F_yf*sin(Uin(2)))+Y(4)*Y(6);
%                      Y(2)*sin(Y(5))+Y(4)*cos(Y(5));
%                 1/m*(F_yf*cos(Uin(2))+F_yr)-Y(2)*Y(6);
%                                     Y(6);
%                       1/I_z*(a*F_yf*cos(Uin(2))-b*F_yr)];

alpha_f = (Uin(2) - atan((Y(4)+1.35*Y(6))/Y(2)));
alpha_r = (- atan((Y(4)-1.45*Y(6))/Y(2)));

psi_yf = ((1-(-1.6))*alpha_f + (-1.6)/0.27*atan(0.27*alpha_f));  % S_hy = 0
psi_yr = ((1-(-1.6))*alpha_r + (-1.6)/0.27*atan(0.27*alpha_r));  % S_hy = 0


F_yf = (1.45/(1.35+1.45)*1400*9.806*0.7*sin(1.35*atan(0.27*psi_yf))); %S_vy = 0;
F_yr = (1.35/(1.35+1.45)*1400*9.806*0.7*sin(1.35*atan(0.27*psi_yr))); %S_vy = 0;

dx = [          Y(2)*cos(Y(5))-Y(4)*sin(Y(5));
           1/1400*(-0.01*1400*9.806+2.00*Uin(1)-F_yf*sin(Uin(2)))+Y(4)*Y(6);
                     Y(2)*sin(Y(5))+Y(4)*cos(Y(5));
                1/1400*(F_yf*cos(Uin(2))+F_yr)-Y(2)*Y(6);
                                    Y(6);
                      1/2667*(1.35*F_yf*cos(Uin(2))-1.45*F_yr)];
                  
end


function [pd] = statepart_hand(Y_vec,U_in)

%Initialize Variables.
m = 1400; N_w = 2.00; f = 0.01; I_z = 2667; a = 1.35; b = 1.45; B_y = 0.27;
C_y = 1.2; D_y = 0.70; E_y = -1.6; g = 9.806;

%Set Variables in Interest.
X = Y_vec(1); u = Y_vec(2); Y = Y_vec(3); v = Y_vec(4); psi = Y_vec(5); r = Y_vec(6);
Fx = U_in(1); delta_f = U_in(2);

%Partial Derivatives of alpha_f and alpha_r.
temp1 = -1/(1 + ((v+a*r)/u)^2); temp2 = -1/(1 + ((v-b*r)/u)^2);
dalpha_f = [0, temp1*-(v+a*r)/u^2, 0, temp1/u, 0, temp1*a/u];
dalpha_r = [0, temp2*-(v-b*r)/u^2, 0, temp2/u, 0, temp2*-b/u];

%Partial Derivatives of phi_yf and phi_yr.
alpha_f = delta_f - atan((v+a*r)/u); alpha_r = - atan((v-b*r)/u);
temp3 = E_y/(1 + (B_y*alpha_f)^2); temp4 = E_y/(1 + (B_y*alpha_r)^2);
dphi_yf = [0, (1 - E_y + temp3)*dalpha_f(2), 0, (1 - E_y + temp3)*dalpha_f(4) ...
    0, (1 - E_y + temp3)*dalpha_f(6)];
dphi_yr = [0, (1 - E_y + temp4)*dalpha_r(2), 0, (1 - E_y + temp4)*dalpha_r(4) ...
    0, (1 - E_y + temp4)*dalpha_r(6)];

%Partial Derivatives of F_yf and F_yr.
phi_yf = (1 - E_y)*alpha_f + E_y*atan(B_y*alpha_f)/B_y;
phi_yr = (1 - E_y)*alpha_r + E_y*atan(B_y*alpha_r)/B_y;
F_zf = b*m*g/(a+b); F_yf = F_zf*D_y*sin(C_y*atan(B_y*phi_yf));
F_zr = a*m*g/(a+b); F_yr = F_zr*D_y*sin(C_y*atan(B_y*phi_yr));
temp5 = F_zf*D_y*cos(C_y*atan(B_y*phi_yf))*C_y*(1/(1+(B_y*phi_yf)^2))*B_y;
temp6 = F_zr*D_y*cos(C_y*atan(B_y*phi_yr))*C_y*(1/(1+(B_y*phi_yr)^2))*B_y;
dF_yf = [0, temp5*dphi_yf(2), 0, temp5*dphi_yf(4), 0, temp5*dphi_yf(6)];
dF_yr = [0, temp6*dphi_yr(2), 0, temp6*dphi_yr(4), 0, temp6*dphi_yr(6)];

%Finally, Compute Partial Derivatives for Jacobian Matrix.
pd = zeros(6);
temp7 = a*cos(delta_f)/I_z;
pd(1,:) = [0, cos(psi), 0, -sin(psi), -u*sin(psi)-v*cos(psi), 0];
pd(2,:) = [0, -sin(delta_f)/m*dF_yf(2), 0, -sin(delta_f)/m*dF_yf(4)+r, ...
    0, -sin(delta_f)/m*dF_yf(6)+v];
pd(3,:) = [0, sin(psi), 0, cos(psi), u*cos(psi)-v*sin(psi), 0];
pd(4,:) = [0, (cos(delta_f)*dF_yf(2)+dF_yr(2))/m - r, 0, ...
    (cos(delta_f)*dF_yf(4)+dF_yr(4))/m, 0, (cos(delta_f)*dF_yf(6)+dF_yr(6))/m - u];
pd(5,:) = [0, 0, 0, 0, 0, 1];
pd(6,:) = [0, temp7*dF_yf(2) - b*dF_yr(2)/I_z, 0, temp7*dF_yf(4) - b*dF_yr(4)/I_z, ...
    0, temp7*dF_yf(6) - b*dF_yr(6)/I_z];
end

function [ip] = inputpart_hand(Y_vec,U_in)
    
%Initialize Variables.
m = 1400; N_w = 2.00; f = 0.01; I_z = 2667; a = 1.35; b = 1.45; B_y = 0.27;
C_y = 1.2; D_y = 0.70; E_y = -1.6; g = 9.806;

%Set Variables in Interest.
X = Y_vec(1); u = Y_vec(2); Y = Y_vec(3); v = Y_vec(4); psi = Y_vec(5); r = Y_vec(6);
Fx = U_in(1); delta_f = U_in(2);

%Partial Derivatives of phi_yf.
alpha_f = delta_f - atan((v+a*r)/u); alpha_r = - atan((v-b*r)/u);
dphi_yf = [0, 1 - E_y + E_y/(1+(B_y*alpha_f)^2)];

%Partial Derivatives of F_yf.
phi_yf = (1 - E_y)*alpha_f + E_y*atan(B_y*alpha_f)/B_y;
F_zf = b*m*g/(a+b); F_yf = F_zf*D_y*sin(C_y*atan(B_y*phi_yf));
dF_yf = [0, F_zf*D_y*cos(C_y*atan(B_y*phi_yf))*C_y*(1/(1+(B_y*phi_yf)^2))*B_y*dphi_yf(2)];

%Finally, Compute Partial Derivatives for Jacobian Matrix.
ip = zeros(6,2);
ip(2,:) = [N_w/m, -sin(delta_f)*dF_yf(2)/m - cos(delta_f)*F_yf/m];
ip(4,:) = [0, a*cos(delta_f)*dF_yf(2)/I_z - a*F_yf*sin(delta_f)/I_z];
ip(6,:) = [0, cos(delta_f)*dF_yf(2)/m - F_yf*sin(delta_f)/m];
end

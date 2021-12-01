%Team 19 Pt 1 Submission

%Strategy:
%Adjust prediction horizon to be a function of braking time to get to a speed where we can handle any turn radius
%Adjust output positon constraints to be the y track constraints as a function of our current x value
%Adjust steering input constraints to be a small deviation relative to the centerline angle
%Use the centerline as a reference trajectory
%Rather than use total steps, use a while loop and end the simulation when the x value of the finish line has been achieved

%% Load Test Track
TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
theta = TestTrack.TestTrack.theta;

%% Define Vehicle Parameters
%Deal with Ftotal > Fmax condition

%number of states and inputs in dynamic model
nstates = 6;
ninputs = 2;

deltaCons = [-0.5 , 0.5];
FxCons = [-5000 , 5000];

input_range = [FxCons ; deltaCons];

mass = 1400;
Nw = 2;
f = 0.01;
Iz = 2667;
a = 1.35;
b = 1.45;
By = 0.27;
Cy = 1.2; % CHECK
Dy = 0.7;
Ey = -1.6;
Shy = 0;
Svy = 0;
g = 9.806;
Fmax = 0.7 * mass * g;

%% Define upper, lower boundaries Attempt 1
%redefine steps as function of total time

dt = 0.01;
inputSteps=246;
tTotal = 60 * 10; %seconds
nsteps = round(tTotal / dt);
stepSize = 1/nsteps;
stepTime = round(tTotal / inputSteps);

d = 5;

ubX = cline(1,:) - d * cos(pi / 2 - theta);
ubX = spline(1:length(ubX),ubX,linspace(1,246,nsteps));
ubY = cline(2,:) + d * sin(pi / 2 - theta);
ubY = spline(1:length(ubY),ubY,linspace(1,246,nsteps));

lbX = cline(1,:) + d * cos(pi / 2 - theta);
lbX = spline(1:length(lbX),lbX,linspace(1,246,nsteps));
lbY = cline(2,:) - d * sin(pi / 2 - theta);
lbY = spline(1:length(lbY),lbY,linspace(1,246,nsteps));

options = optimoptions('fmincon','SpecifyConstraintGradient',true,...
                       'SpecifyObjectiveGradient',true) ;
                   
ub = zeros(8*nsteps-2,1);

ub(1:6:6*nsteps) = ubX ;
ub(2:6:6*nsteps) = Inf ;
ub(3:6:6*nsteps) = ubY ;
ub(4:6:6*nsteps) = Inf ;
ub(5:6:6*nsteps) = pi;
ub(6:6:6*nsteps) = Inf ;
ub(6*nsteps+1:2:8*nsteps-2) = 0.5 ;
ub(6*nsteps+2:2:8*nsteps-2) = 5000 ;

lb = zeros(8*nsteps-2,1);

lb(1:6:6*nsteps) =  lbX;
lb(2:6:6*nsteps) = -Inf ;
lb(3:6:6*nsteps) = lbY ;
lb(4:6:6*nsteps) = -Inf ;
lb(5:6:6*nsteps) = -pi ;
lb(6:6:6*nsteps) = -Inf ;
lb(6*nsteps+1:2:8*nsteps-2) = -0.5 ;
lb(6*nsteps+2:2:8*nsteps-2) = -5000 ;

%% Bounds, Attempt 2

%Interpolation

interp_size = 10;
bl = [interp(bl(1,:),interp_size);interp(bl(2,:),interp_size)];
br = [interp(br(1,:),interp_size);interp(br(2,:),interp_size)];
theta = interp(theta,interp_size);
nsteps = size(bl,2);

lowbounds = min(bl,br);
highbounds = max(bl,br);

[lb,ub]=bound_cons(nsteps,theta ,lowbounds, highbounds);

%% Run optimization
x0 = [287,5,-176,0,2,0];
X0 = [repmat(x0 , 1 , nsteps) , repmat([0,0] , 1 , nsteps-1)];

cf=@costfun;
nc=@nonlcon;

z = fmincon(cf,X0,[],[],[],[],lb',ub',nc,options);

%% Generate U

U = reshape(z(6*nsteps+1:end),2,nsteps-1);
%U=@(t) [interp1(0:dt:(nsteps - 2)*dt,U(1,:),t,'previous','extrap');...
       % interp1(0:dt:(nsteps - 2)*dt,U(2,:),t,'previous','extrap')];

%% Test Plot

[Y,T] = forwardIntegrateControlInput(U);

figure(1)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(Y(:,1),Y(:,3),'r')
hold off

%% Functions (taken from GitHub)

function [lb,ub]=bound_cons(nsteps,theta ,lowbounds, highbounds) %,input_range

ub = [];
lb = [];

for i = 1:nsteps
    
ub = [ub,[highbounds(1,i), +inf,highbounds(2,i), +inf, theta(i)+pi/2, +inf]];

lb = [lb,[lowbounds(1,i), -inf, lowbounds(2,i), -inf, theta(i)-pi/2, -inf]];

end

ub = [ub,repmat([5000,0.5],1,nsteps-1) ]';
lb = [lb,repmat([-5000,-0.5],1,nsteps-1) ]';

end

function [dx] = odefun(Y,Uin)

m    = 1400;                % Mass of Car
Nw  = 2.00;                % ?? 
f    = 0.01;                % ??
Iz  = 2667;                 % Momemnt of Inertia
a    = 1.35;                % Front Axle to COM
b    = 1.45;                % Rear Axle to Com
By  = 0.27;                 % Empirically Fit Coefficient
Cy  = 1.2;                  % Empirically Fit Coefficient
Dy  = 0.70;                 % Empirically Fit Coefficient
Ey  = -1.6;                 % Empirically Fit Coefficient
g    = 9.806;               % Graviational Constant

alpha_f = (Uin(2) - atan((Y(4)+a*Y(6))/Y(2)));
alpha_r = (- atan((Y(4)-b*Y(6))/Y(2)));

psi_yf = ((1-(Ey))*alpha_f + (Ey)/By*atan(By*alpha_f));  % S_hy = 0
psi_yr = ((1-(Ey))*alpha_r + (Ey)/By*atan(By*alpha_r));  % S_hy = 0


F_yf = (b/(a+b)*m*g*Dy*sin(Cy*atan(By*psi_yf))); %S_vy = 0;
F_yr = (a/(a+b)*m*g*Dy*sin(Cy*atan(By*psi_yr))); %S_vy = 0;

dx = [Y(2)*cos(Y(5))-Y(4)*sin(Y(5));
      1/m*(-f*m*g+Nw*Uin(1)-F_yf*sin(Uin(2)))+Y(4)*Y(6);
      Y(2)*sin(Y(5))+Y(4)*cos(Y(5));
      1/m*(F_yf*cos(Uin(2))+F_yr)-Y(2)*Y(6);
      Y(6);
      1/Iz*(a*F_yf*cos(Uin(2))-b*F_yr)];
                  
end

function [J, dJ] = costfun(z)
    if size(z,2) > size(z,1)
        z = z' ;
    end
    nsteps = (size(z,1)+2)/8 ;

    zx = z(1:6*nsteps) ;
    zu = z(6*nsteps+1:end) ;
    R=eye(2*nsteps-2);

    nom=zeros(6*nsteps,1) ;
    nom(1:6:6*nsteps) = 1472 ;
    nom(3:6:6*nsteps+2) = 818 ;
    Q = diag(repmat([1,0,1,0,1,0],1,nsteps));

    J = zu'*R*zu+(zx-nom)'*Q*(zx-nom) ;
    dJ = [2*Q*zx-2*Q*nom;
          2*R*zu]' ;
end

function [g,h,dg,dh] = nonlcon(z)
x0   =   287;
u0   =   5.0;
y0   =  -176;
v0   =   0.0;
psi0 =   2.0;
r0   =   0.0;

z0 = [x0, u0, y0, v0, psi0, r0];

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
            h(1:6) = z(1:6,:) - z0' ;
            dh(1:6,1:6) = eye(6) ; 
        else
            h(6*i-5:6*i) = zx(6*i-5:6*i)-zx(6*i-11:6*i-6)-...
                               dt*odefun(zx(6*i-11:6*i-6),zu(2*i-3:2*i-2)) ;
                           
            dh(6*i-5:6*i,6*i-11:6*i) = [-eye(6)-dt*statepart_hand(zx(6*i-11:6*i-6),zu(2*i-3:2*i-2)),eye(6)] ;
            dh(6*i-5:6*i,6*nsteps+2*i-3:6*nsteps+2*i-2) = -dt*inputpart_hand(zx(6*i-11:6*i-6),zu(2*i-3:2*i-2)) ;
        end
        
    end

    dg = dg' ;
    dh= dh' ;
end

function [pd] = statepart_hand(Y_vec,U_in)

%Initialize Variables.
m = 1400; 
N_w = 2.00;
f = 0.01;
I_z = 2667;
a = 1.35;
b = 1.45;
B_y = 0.27;
C_y = 1.2;
D_y = 0.70;
E_y = -1.6;
g = 9.806;

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

%Partial Derivatives for Jacobian Matrix.
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

%Partial Derivatives for Jacobian Matrix.
ip = zeros(6,2);
ip(2,:) = [N_w/m, -sin(delta_f)*dF_yf(2)/m - cos(delta_f)*F_yf/m];
ip(4,:) = [0, a*cos(delta_f)*dF_yf(2)/I_z - a*F_yf*sin(delta_f)/I_z];
ip(6,:) = [0, cos(delta_f)*dF_yf(2)/m - F_yf*sin(delta_f)/m];
end
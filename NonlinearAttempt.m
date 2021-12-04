%% Team 19 Controls Part 1
clear all
clc

%% Load Data

load ('TestTrack.mat');

bl = TestTrack.bl;       % Left Boundaries
br = TestTrack.br;       % Right Boundaries
cline = TestTrack.cline; % Center Line
theta = TestTrack.theta; % Center Line's Orientation

dt = 0.01;

%Manually determined U matrix
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
theta_trajec = trajec(5,:);
trajec = trajec([1,3],:);

% figure(1)
% plot(bl(1,:),bl(2,:),'k')
% hold on
% plot(br(1,:),br(2,:),'k')
% plot(trajec(1,:),trajec(2,:),'b--')

%% Initialize Constants

m    = 1400;                % Mass of Car
N_w  = 2.00;                % ?? 
f    = 0.01;                % ??
I_z  = 2667;                % Momemnt of Inertia
a    = 1.35;                % Front Axle to COM
b    = 1.45;                % Rear Axle to Com
B_y  = 0.27;                % Empirically Fit Coefficient
C_y  = 1.2;                 % Empirically Fit Coefficient
D_y  = 0.70;                % Empirically Fit Coefficient
E_y  = -1.6;                % Empirically Fit Coefficient
S_hy = 0.00;                % Horizontal Offset in y
S_vy = 0.00;                % Vertical Offset in y
g    = 9.806;               % Graviational Constant

upperInputCons = [0.5 , 5000];
lowerInputCons = [-0.5 , -5000];

%% Initialize Time and Prediction Data

dt   = 0.01;% Time Step

interp_size = 1; 

%try using spline to have evenly spaced points and the ideal total size

bl = [interp(bl(1,:),interp_size);interp(bl(2,:),interp_size)];
br = [interp(br(1,:),interp_size);interp(br(2,:),interp_size)];
cline = [interp(cline(1,:),interp_size);interp(cline(2,:),interp_size)];
theta = interp(theta,interp_size);

trajec = imresize(trajec,[2,size(cline,2)]);
theta_trajec = imresize(theta_trajec,[2,size(cline,2)]);


%% Define Path Boundaries

d = 5;

ubX = cline(1,:) - d * cos(pi / 2 - theta);
ubY = cline(2,:) + d * sin(pi / 2 - theta);
ub1 = [ubX ; ubY];
lbX = cline(1,:) + d * cos(pi / 2 - theta);
lbY = cline(2,:) - d * sin(pi / 2 - theta);
lb1 = [lbX ; lbY];

% figure(2)
% plot(bl(1,:),bl(2,:),'k')
% hold on
% plot(br(1,:),br(2,:),'k')
% plot(lb1(1,:),lb1(2,:),'r')
% plot(ub1(1,:), ub1(2,:),'r')
% plot(trajec(1,:),trajec(2,:),'b--')

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
u0 = [-0.04 , 5000];
           
%% Path Segmentation: Start
%Update to select new path (based on critical path indicies) within a for loop
nsteps = 25;

i = 1;
blSeg = bl(:,i:i+nsteps-1);
brSeg = br(:,i:i:i+nsteps-1);
clineSeg = cline(:,i:i+nsteps-1);
thetaSeg = theta(:,i:i+nsteps-1) ;
trajecSeg = trajec(:,i:i+nsteps-1);
theta_trajecSeg = theta_trajec(:,i:i+nsteps-1) ;

% lowboundsSeg = lb1(:,i:i+nsteps-1);
% highboundsSeg = ub1(:,i:i+nsteps-1);

lowboundsSeg = min(blSeg,brSeg);
highboundsSeg = max(blSeg,brSeg);

%% Path Generation using NL Optimization

[lb,ub]=bound_cons(nsteps,theta_trajecSeg ,lowboundsSeg, highboundsSeg, upperInputCons, lowerInputCons);


options = optimoptions('fmincon','SpecifyConstraintGradient',true,...
                       'SpecifyObjectiveGradient',true) ;
                   
endpoints = [trajecSeg(1,nsteps),trajecSeg(2,nsteps),theta_trajecSeg(nsteps)]; %change back to reference trajectory after sizes are fixed
xrefs = z0(1):(endpoints(1)-z0(1))/(nsteps-1):endpoints(1);
yrefs = z0(3):(endpoints(2)-z0(3))/(nsteps-1):endpoints(2);
thetarefs = z0(5):(endpoints(3)-z0(5))/(nsteps-1):endpoints(3);

states0 = [xrefs;z0(2)*ones(1,nsteps);yrefs;z0(4)*ones(1,nsteps);thetarefs;z0(6)*ones(1,nsteps)];
states0 = reshape(states0,1,nsteps*nstates);

X0 = [states0, repmat(u0,1,nsteps-1)];

cf = @(z)costfun_segmt(z, trajecSeg, theta_trajecSeg, nsteps);
nc = @(z)nonlcon(z,z0,nstates,ninputs);

z=fmincon(cf,X0,[],[],[],[],lb',ub',nc,options);

Y=reshape(z(1:6*nsteps),6,nsteps)';
z0 = Y(end , :);
U1=reshape(z(6*nsteps+1:end),2,nsteps-1)';
U= [U1;[U1(end,1) , U1(end,2)]];
u0 = U(end,:);

%% For loop attempt
hold off

%i = nsteps;
%while Y(1,end) < cline(1,end) && Y(3,end) < cline(2,end)

for i = nsteps : nsteps : 4*nsteps %(size(cline,2)-nsteps)
   
%     if i > size(cline,2)
%         
%         blSeg = bl(:,end);
%         brSeg = br(:,end);
%         clineSeg = cline(:,end);
%         thetaSeg = theta(:,end) ;
%         trajecSeg = trajec(:,end);
%         theta_trajecSeg = theta_trajec(:,end);
%         
%     else
        
        blSeg = bl(:,i:i+nsteps-1);
        brSeg = br(:,i:i+nsteps-1);
        clineSeg = cline(:,i:i+nsteps-1);
        thetaSeg = theta(:,i:i+nsteps-1) ;
        trajecSeg = trajec(:,i:i+nsteps-1);
        theta_trajecSeg = theta_trajec(:,i:i+nsteps-1) ;
    
    %end

    lowboundsSeg = min(blSeg,brSeg);
    highboundsSeg = max(blSeg,brSeg);

    [lb,ub]=bound_cons(nsteps,theta_trajecSeg ,lowboundsSeg, highboundsSeg, upperInputCons, lowerInputCons);


    options = optimoptions('fmincon','SpecifyConstraintGradient',true,...
                           'SpecifyObjectiveGradient',true) ;

    endpoints = [trajecSeg(1,end),trajecSeg(2,end),theta_trajecSeg(end)];    
    xrefs = z0(1):(endpoints(1)-z0(1))/(nsteps-1):endpoints(1);
    yrefs = z0(3):(endpoints(2)-z0(3))/(nsteps-1):endpoints(2);
    thetarefs = z0(5):(endpoints(3)-z0(5))/(nsteps-1):endpoints(3);

    states0 = [xrefs;z0(2)*ones(1,nsteps);yrefs;z0(4)*ones(1,nsteps);thetarefs;z0(6)*ones(1,nsteps)];
    states0 = reshape(states0,1,nsteps*nstates);

    X0 = [states0, repmat(u0,1,nsteps-1)];

    cf= @(z)costfun_segmt(z, trajecSeg, theta_trajecSeg, (i - 1) + nsteps);
    nc=@(z)nonlcon(z,z0,nstates,ninputs); %readd z0 as second entry if broken

    z=fmincon(cf,X0,[],[],[],[],lb',ub',nc,options);

    Ytemp=reshape(z(1:6*nsteps),6,nsteps)';
    z0 = Ytemp(end , :);
    Y = [Y ; Ytemp];
    
    Utemp=reshape(z(6*nsteps+1:end),2,nsteps-1)';
    Utemp=[Utemp;[Utemp(end,1) , Utemp(end,2)]];
    u0 = Utemp(end,:);
    U = [U ; Utemp];
     
%     figure(5)
%     plot(bl(1,:),bl(2,:),'k')
%     plot(br(1,:),br(2,:),'k')
%     plot(blSeg(1,:),blSeg(2,:),'r')
%     hold on
%     plot(brSeg(1,:),brSeg(2,:),'r')
%     plot(Ytemp(:,1),Ytemp(:,3),'b')
%     plot(cline(1,:) , cline(2,:),'k:')
%     plot(endpoints(1) , endpoints(2) , 'o')
    
end
hold off

%% Plotting

figure(3)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(Y(:,1),Y(:,3),'b')
hold off


% plot(highbounds(1,:) , highbounds(2,:) , 'r')
% plot(lowbounds(1,:) , lowbounds(2,:) , 'r')


%% Test area

[g,h,dg,dh]=nonlcon(z , x0 , nstates,ninputs);
find( h~= 0 )
    
%% Functions

function [lb,ub]=bound_cons(nsteps, theta_trajec ,lowbounds, highbounds,upperInputCons,lowerInputCons) %,input_range

ub = [];
lb = [];

    for i = 1:nsteps
    
        ub = [ub,[highbounds(1,i), +inf, highbounds(2,i), +inf, theta_trajec(i)+pi/3 , +inf]]; %theta_trajec(i)+pi/3
        lb = [lb,[lowbounds(1,i), -inf, lowbounds(2,i), -inf, theta_trajec(i)-pi/3 , -inf]]; %theta_trajec(i)-pi/3

    end

ub = [ub,repmat(upperInputCons,1,nsteps-1) ]';
lb = [lb,repmat(lowerInputCons,1,nsteps-1) ]';

end

function [J, dJ] = costfun_segmt(z,trajec,theta_trajec, nsteps)
    if size(z,2) > size(z,1)
        z = z' ;
    end
    nsteps = (size(z,1)+2)/8 ;

    zx = z(1:6*nsteps) ;
    zu = z(6*nsteps+1:end) ;

    R = eye(2*nsteps-2);

    nom=zeros(6*nsteps,1) ;
    nom(1:6:6*nsteps) =  trajec(1,nsteps) ; %REPLACE W X IDX 
    nom(3:6:6*nsteps+2) = trajec(2,nsteps) ;
    nom(5:6:6*nsteps+4) = theta_trajec(nsteps) ;

    Q = diag(repmat([1,0,1,0,1,0],1,nsteps));  %OPTIMIZE TO MAXIMIZE SPEED

    J = zu'*R*zu+(zx-nom)'*Q*(zx-nom) ;
    dJ = [2*Q*zx-2*Q*nom;...
          2*R*zu]' ;
end

function [g,h,dg,dh]=nonlconP3(z,nstates,ninputs)
    
    ntotal = nstates+ninputs;

    if size(z,2)>size(z,1)
        z = z' ;
    end
    nsteps = (size(z,1)+ninputs)/ntotal ;
%     b = 1.5 ; 
%     L = 3 ;
%     dt = 0.05 ;

    dt = 0.01;

    zx=z(1:nsteps*nstates);
    zu=z(nsteps*nstates+1:end);

    g = zeros(nsteps,1) ;
    dg = zeros(nsteps,5*nsteps-2) ;

    h = zeros(nstates*nsteps,1) ;
    dh = zeros(nstates*nsteps,5*nsteps-2);

    h(1:nstates) = z(1:nstates,:) ;
    dh(1:nstates,1:nstates)=eye(nstates);

    for i=1:nsteps
        if i==1
            h(1:nstates) = z(1:nstates,:);
            dh(1:nstates,1:nstates)=eye(nstates); 
        else
            h(3*i-2:3*i) = zx(3*i-2:3*i)-zx(3*i-5:3*i-3)-dt*odefun(zx(3*i-5:3*i-3),zu(2*i-3:2*i-2)) ;
            dh(3*i-2:3*i,3*i-5:3*i) = [-eye(3)-dt*statepart(zx(3*i-5:3*i-3),zu(2*i-3:2*i-2)),eye(3)] ;
            dh(3*i-2:3*i,3*nsteps+2*i-3:3*nsteps+2*i-2) = -dt*inputpart(zx(3*i-5:3*i-3),zu(2*i-3:2*i-2)) ;
        end

    end

    dg = dg' ;
    dh= dh' ;
end

function [g,h,dg,dh]=nonlcon(z , x0 , nstates,ninputs)

ntotal = nstates+ninputs;

    if size(z,2) > size(z,1)
        z = z' ;
    end
    
    nsteps = (size(z,1)+ninputs)/ntotal ;
    dt = 0.01 ;

    zx = z(1:nsteps*nstates) ;
    zu = z(nsteps*ninputs+1:end) ;
    
    g = [];
    dg = [];
    
    h = zeros(6*nsteps,1) ;
    dh = zeros(6*nsteps,8*nsteps-2);

    for i = 1:nsteps
        
        if i == 1
            h(1:nstates) = z(1:nstates,:) - x0' ;
            dh(1:nstates,1:nstates)=eye(nstates);
            
%             h(1:6) = z(1:6,:) - x0' ;
%             dh(1:6,1:6) = eye(6) ; 
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
F_zf = b*m*g/(a+b); 
F_yf = F_zf*D_y*sin(C_y*atan(B_y*phi_yf));
F_zr = a*m*g/(a+b); 
F_yr = F_zr*D_y*sin(C_y*atan(B_y*phi_yr));
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

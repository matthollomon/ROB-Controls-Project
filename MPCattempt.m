%% Reverted MPC Attempt

%% Load track data

TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
thetaCline = TestTrack.TestTrack.theta;

%% Define reference states, inputs

Umanual = [repmat([-0.05, 100],50,1);
    repmat([-0.1, 200],50,1);
    repmat([-0.1, 300],50,1);
    repmat([-0.05, 400],50,1);
    repmat([-0.0, 500],50,1);
    repmat([0.05, 600],50,1);
    repmat([0.05, 700],50,1);
    repmat([0.05, 800],50,1);
    repmat([-0.0, 900],50,1);
    repmat([0, 1000],50,1);
    repmat([0, 900],50,1);
    repmat([0, 800],50,1);
    repmat([0, 700],50,1);
    repmat([0, 600],50,1);
    repmat([0, 500],50,1);
    repmat([0, 400],50,1);
    repmat([0, 300],50,1);
    repmat([0, 200],50,1);
    repmat([0, 100],50,1);
    repmat([0, 0],600,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0, 0],1100,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([-.01, 0],50,1);
    repmat([-.01, 0],50,1);
    repmat([-.01, 0],50,1);
    repmat([-.01, 0],50,1);
    repmat([-.01, 0],50,1);
    repmat([-.02, 0],50,1);
    repmat([-.03, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.05, 0],50,1);
    repmat([-.05, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.05, 0],50,1);
    repmat([-.05, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.03, 0],50,1);
    repmat([-.02, 0],50,1);
    repmat([-.01, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    
    
    
    ];

dt = 0.01;
time = size(Umanual,1) * dt;
T = 0 : dt : time;
    
x0 = [287 , 5 , -176 , 0 , 2 , 0];
[Ymanual,Tmanual] = forwardIntegrateControlInput(Umanual,x0);

Y_ref = Ymanual';

U_ref = [Umanual(:,2)'; Umanual(:,1)'];

%% Parameters

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
Cy = 1.2; 
Dy = 0.7;
Ey = -1.6;
Shy = 0;
Svy = 0;
g = 9.806;
Fmax = 0.7 * mass * g;

%% Generate A, B matricies
dt = 0.01;

[A,B,f] = DiscretizedLinearizedModel( Y_ref, U_ref,dt);


%% Number of decision variables

npred=10;

Ndec=(npred+1)*nstates+ninputs*npred;

%% Test EQ, Boundary constraints at index 1

[Aeq_test1,beq_test1] = eq_cons(1,A,B,x0,npred,nstates,ninputs);
[Lb_test1,Ub_test1] = bound_cons(1,U_ref,npred,input_range,nstates,ninputs);

%% Simulate controller
%initial condition
y0 = [287,5,-176,0,2,0]';

%final trajectory
Y = NaN(nstates,length(T)-1);
Y(:,1) = y0;

%applied inputs
U = NaN(ninputs,length(T)-1);
U(:,1) = U_ref(:,1);

%input from QP
u_mpc = NaN(ninputs,length(T)-1);

%error in states (actual-reference)
eY = NaN(nstates,length(T)-1);

aQ = 1;
bQ = 1;
testLength = length(T) - 2; 

for i = 1 : testLength
    %shorten prediction horizon if we are at the end of trajectory
    %lengthen prediction horizon if going fast
    %shorten if going slow
    npred_i=min([npred,length(T) - 1 - i]);
    
    %calculate error
    eY(:,i) = Y(:,i) - Y_ref(:,i);

    %generate equality constraints USE REF TRAJECTORY RATHER THAN ERROR??
    [Aeq,beq] = eq_cons(i,A,B,eY(:,i),npred_i,nstates,ninputs);
    
    %generate boundary constraints
    [Lb,Ub] = bound_cons(i,U_ref,npred_i,input_range,nstates,ninputs);
    
    %cost = Sum from i to npred+1 (a(ey(1,i))^2 + b(ey(3,i))^2 + 0(nothing
    %for inputs)
       
    fdisc = zeros(nstates*(npred_i+1)+ninputs*npred_i,1);
    
%     fdisc = [];
%     for k = i : npred_i + i
%       fdisc = [fdisc;
%                f(k)];     
%     end
%     
%     %What should f for inputs be?
%     for k = i : npred_i + i - 1
%     fdisc = [fdisc;            
%              0;
%              0];
%     end
    
%    %H = df^2/d^2z
    
%     H = diag([Q,repmat(R,[1,npred_i])]);

    %cost for states OPTIMIZE
    
    Q=[1,0.01,1,0.01,1,0.01];
    
    %cost for inputs OPTIMIZE
    R=[10,10];
    
    H=diag([repmat(Q,[1,npred_i+1]),repmat(R,[1,npred_i])]);
    
    [x,fval] = quadprog(H,fdisc,[],[],Aeq,beq,Lb,Ub);
    
    
    %get linearized input
    u_mpc(:,i) = x(nstates*(npred_i+1)+1:nstates*(npred_i+1)+ninputs);
    
    %get input
    U(:,i) = u_mpc(:,i) + U_ref(:,i);
    
    %simulate model
    [~,ztemp]=ode45(@(t,x)bike(t,x,0,U(:,i)),[0 dt], Y(:,i));
    
    %store final state
    Y(:,i+1) = ztemp(end,:)';
   
end

%% Plotting

Ucheck = [U(2,:)', U(1,:)'];
[Ycheck,T] = forwardIntegrateControlInput(Ucheck);

figure(1)
%plot(Y(1,1:testLength),Y(3,1:testLength),'c')
plot(Ycheck(:,1),Ycheck(:,3),'r')
hold on
plot(Ymanual(:,1),Ymanual(:,3),'b')
plot(bl(1,:),bl(2,:),'k')
plot(br(1,:),br(2,:),'k')
%plot(cline(1,:),cline(2,:),'--k')
legend('MPC trajectory','Manual Trajectory')
xlim([200 400])
ylim([-200 200])
hold off

figure(2)
subplot(2,1,1)
title('Inputs')
plot(T(1:1:testLength),U(2,1:testLength),'r')
hold on
plot(T(1:1:testLength),U_ref(2,1:testLength),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
ylabel('Steering Angle [rad]')
legend('MPC','Reference')
ylim([-0.5 0.5])
hold off

subplot(2,1,2)
plot(T(1:1:testLength),U(1,1:testLength),'r')
hold on
plot(T(1:1:testLength),U_ref(1,1:testLength),'b--')
%plot(T(1:1:testLength),u_mpc(1,1:testLength),'k.-')
ylabel('Acceleration [m/s^2]')
legend('MPC','Reference')
ylim([-5000 5000])
hold off

figure(3)
title('States')
subplot(2,3,1)
plot(T(1:1:testLength),Ycheck(1:testLength,1),'r')
hold on
plot(Tmanual(1:1:testLength),Ymanual(1:testLength,1),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
title('x')
legend('MPC','Reference')
hold off

subplot(2,3,4)
plot(T(1:1:testLength),Ycheck(1:testLength,2),'r')
hold on
plot(Tmanual(1:1:testLength),Ymanual(1:testLength,2),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
title('u')
legend('MPC','Reference')
hold off

subplot(2,3,2)
plot(T(1:1:testLength),Ycheck(1:testLength,3),'r')
hold on
plot(Tmanual(1:1:testLength),Ymanual(1:testLength,3),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
title('y')
legend('MPC','Reference')
hold off

subplot(2,3,5)
plot(T(1:1:testLength),Ycheck(1:testLength,4),'r')
hold on
plot(Tmanual(1:1:testLength),Ymanual(1:testLength,4),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
title('v')
legend('MPC','Reference')
hold off

subplot(2,3,3)
plot(T(1:1:testLength),Ycheck(1:testLength,5),'r')
hold on
plot(Tmanual(1:1:testLength),Ymanual(1:testLength,5),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
title('psi')
legend('MPC','Reference')
hold off

subplot(2,3,6)
plot(T(1:1:testLength),Ycheck(1:testLength,6),'r')
hold on
plot(Tmanual(1:1:testLength),Ymanual(1:testLength,6),'b--')
%plot(T(1:1:testLength),u_mpc(2,1:testLength),'k.-')
title('r')
legend('MPC','Reference')
hold off


%% Functions

% INCLUDE i EVERYWHERE IT SHOULD APPEAR
function [A, B, f] = DiscretizedLinearizedModel(Xbar_k, Ubar_k,dt)
%Ts = sampling time
m = 1400;
g = 9.806;
Nw = 2;
fcon = 0.01;
Iz = 2667;
a = 1.35;
b = 1.45;
By = 0.27;
Cy = 1.2;
Dy = 0.7;
Ey = -1.6;
Fzf = b/(a+b)*m*g;
Fzr = a/(a+b)*m*g;
sx = 6;
su = 2;

u = @(i) Xbar_k(2,i);
r = @(i) Xbar_k(6,i);
v = @(i) Xbar_k(4,i);
psi = @(i) Xbar_k(5,i);
deltaf = @(i) Ubar_k(2,i);
Fx = @(i) Ubar_k(1,i);

dafddeltaf = 180.0/pi;
dafdv = @(i) rad2deg(-1/(u(i)*((v(i) + a*r(i))^2/u(i)^2 + 1)));
dafdr = @(i) rad2deg(-a/(u(i)*((v(i) + a*r(i))^2/u(i)^2 + 1)));
dafdu = @(i) rad2deg((v(i) + a*r(i))/(u(i)^2*((v(i) + a*r(i))^2/u(i)^2 + 1)));

dardv = @(i) rad2deg(-1/(u(i)*((v(i) - b*r(i))^2/u(i)^2 + 1)));
dardr = @(i) rad2deg(b/(u(i)*((v(i) - b*r(i))^2/u(i)^2 + 1)));
dardu = @(i) rad2deg((v(i) - b*r(i))/(u(i)^2*((v(i) - b*r(i))^2/u(i)^2 + 1)));

af = @(i) rad2deg(deltaf(i) - atan2(v(i)+a*r(i), u(i)));
ar = @(i) rad2deg(-atan2(v(i)-b*r(i), u(i)));

phiyf = @(i) (1 - Ey)*af(i) + Ey/By*atan(By*af(i));
phiyr = @(i) (1 - Ey)*ar(i) + Ey/By*atan(By*ar(i));

dphiyfdaf = @(i) Ey/(By^2*af(i)^2 + 1) - Ey + 1;
dphiyrdar = @(i) Ey/(By^2*ar(i)^2 + 1) - Ey + 1;

Fyf = @(i) Fzf*Dy*sin(Cy*atan(By*phiyf(i)));
Fyr = @(i) Fzr*Dy*sin(Cy*atan(By*phiyr(i)));

dFyfdphiyf = @(i) (By*Cy*Dy*Fzf*cos(Cy*atan(By*phiyf(i))))/(By^2*phiyf(i)^2 + 1);
dFyrdphiyr = @(i) (By*Cy*Dy*Fzr*cos(Cy*atan(By*phiyr(i))))/(By^2*phiyr(i)^2 + 1);
dFyfdaf = @(i) dFyfdphiyf(i)*dphiyfdaf(i);
dFyrdar = @(i) dFyrdphiyr(i)*dphiyrdar(i);

dFyfdu = @(i) dFyfdaf(i)*dafdu(i);
dFyrdu = @(i) dFyrdar(i)*dardu(i);

dFyfdv = @(i) dFyfdaf(i)*dafdv(i);
dFyrdv = @(i) dFyrdar(i)*dardv(i);

dFyfdr = @(i) dFyfdaf(i)*dafdr(i);
dFyrdr = @(i) dFyrdar(i)*dardr(i);
dFyfddeltaf = @(i) dFyfdaf(i)*dafddeltaf;

A = @(i) eye(6) + dt*[ 0, 0, 0, 0, 0, 0; ...
                        cos(psi(i)), -1.0/m*dFyfdu(i)*sin(deltaf(i)), sin(psi(i)), 1.0/m*(dFyfdu(i)*cos(deltaf(i)) + dFyrdu(i)) - r(i), 0, 1.0/Iz*(a*dFyfdu(i)*cos(deltaf(i)) - b*dFyrdu(i)); ...
                        0, 0, 0, 0, 0, 0; ...
                        -sin(psi(i)), -1.0/m*dFyfdv(i)*sin(deltaf(i)) + r(i), cos(psi(i)), 1.0/m*(dFyfdv(i)*cos(deltaf(i)) + dFyrdv(i)), 0, 1.0/Iz*(a*dFyfdv(i)*cos(deltaf(i)) - b*dFyrdv(i)); ...
                        -u(i)*sin(psi(i)) - v(i)*cos(psi(i)), 0, u(i)*cos(psi(i)) - v(i)*sin(psi(i)), 0, 0, 0; ...
                        0, -1.0/m*dFyfdr(i)*sin(deltaf(i)) + v(i), 0, 1.0/m*(dFyfdr(i)*cos(deltaf(i)) + dFyrdr(i)) - u(i), 1.0, 1.0/Iz*(a*dFyfdr(i)*cos(deltaf(i)) - b*dFyrdr(i))]';

B = @(i) dt*[ 0, -1.0/m*(dFyfddeltaf(i)*sin(deltaf(i)) + Fyf(i)*cos(deltaf(i))), 0, 1.0/m*(dFyfddeltaf(i)*cos(deltaf(i)) - Fyf(i)*sin(deltaf(i))), 0, a/Iz*(dFyfddeltaf(i)*cos(deltaf(i)) - Fyf(i)*sin(deltaf(i))); ...
                0, Nw/m, 0, 0, 0, 0]';

f = @(i) [u(i)*cos(psi(i))-v(i)*sin(psi(i));
     1/m*(-fcon*m*g+Nw*Fx(i)-Fyf(i)*sin(deltaf(i)))+v(i)*r(i);
     u(i)*sin(psi(i))+v(i)*cos(psi(i));
     1/m*(Fyf(i)*cos(deltaf(i))+Fyr(i))-u(i)*r(i);
     r(i);
     1/Iz*(a*Fyf(i)*cos(deltaf(i))-b*Fyr(i))];
 
% gc= @(i) f-Ac*Xbar_k(1:sx)-Bc*Ubar_k(1:su);
% 
% Bc_aug=[Bc gc];

%discretize

% % see report for proof of following method
% tmp = expm([Ac Bc_aug; zeros(su+1,sx+su+1)]*Ts);
% 
% Ad = zeros(sx+1,sx+1);
% Bd = zeros(sx+1,su+1);
% gd = zeros(sx+1,1);
% Ad(1:sx,1:sx) =tmp(1:sx,1:sx);
% Bd(1:sx,1:su) =tmp(1:sx,sx+1:sx+su);
% gd(1:sx) =tmp(1:sx,sx+su+1);
% 
% % following to avoid numerical errors
% Ad(end,end)=1;
% Bd(end,end)=Ts;


end

function [Aeq,beq]=eq_cons(initial_idx,A,B,x_initial,npred,nstates,ninputs)
%build matrix for A_i*x_i+B_i*u_i-x_{i+1}=0
%in the form Aeq*z=beq
%initial_idx specifies the time index of initial condition from the reference trajectory 
%A and B are function handles above

%initial condition
x_initial=x_initial(:);

%size of decision variable and size of part holding states
zsize=(npred+1)*nstates+npred*ninputs;
xsize=(npred+1)*nstates;

Aeq=zeros(xsize,zsize);
Aeq(1:nstates,1:nstates)=eye(nstates); %initial condition 
beq=zeros(xsize,1);
beq(1:nstates)=x_initial;

state_idxs=nstates+1:nstates:xsize;
input_idxs=xsize+1:ninputs:zsize;

for i=1:npred
    %negative identity for i+1
    Aeq(state_idxs(i):state_idxs(i)+nstates-1,state_idxs(i):state_idxs(i)+nstates-1)=-eye(nstates);
    
    %A matrix for i
    Aeq(state_idxs(i):state_idxs(i)+nstates-1,state_idxs(i)-nstates:state_idxs(i)-1)=A(initial_idx+i-1);
    
    %B matrix for i
    Aeq(state_idxs(i):state_idxs(i)+nstates-1,input_idxs(i):input_idxs(i)+ninputs-1)=B(initial_idx+i-1);
end

end

function [Lb,Ub]=bound_cons(initial_idx,U_ref,npred,input_range,nstates,ninputs)
%time_idx is the index along uref the initial condition is at
xsize=(npred+1)*nstates;
usize=npred*ninputs;

Lb=[];
Ub=[];

for j=1:ninputs
    Lb=[Lb;input_range(j,1)-U_ref(j,initial_idx:initial_idx+npred-1)];
    Ub=[Ub;input_range(j,2)-U_ref(j,initial_idx:initial_idx+npred-1)];
end

Lb=reshape(Lb,[usize,1]);
Ub=reshape(Ub,[usize,1]);

Lb=[-Inf(xsize,1);Lb];
Ub=[Inf(xsize,1);Ub];

end

function dzdt=bike(t,x,T,U_in)
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


%generate input functions CHANGE TO REFLECT ACTUAL U SIZE 
if length(T)<=1 || isempty(T) || size(U_in,2)==1
    delta_f=U_in(1);
    F_x=U_in(2);
else
    delta_f=interp1(T',U_in(1,:)',t,'previous');
    F_x=interp1(T',U_in(2,:)',t,'previous');
end

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
dzdt= [x(2)*cos(x(5))-x(4)*sin(x(5));...
          (-f*m*g+Nw*F_x-F_yf*sin(delta_f))/m+x(4)*x(6);...
          x(2)*sin(x(5))+x(4)*cos(x(5));...
          (F_yf*cos(delta_f)+F_yr)/m-x(2)*x(6);...
          x(6);...
          (F_yf*a*cos(delta_f)-F_yr*b)/Iz];
end
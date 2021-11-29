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
thetaCline = TestTrack.TestTrack.theta;

%% Define Vehicle Parameters
%Deal with Ftotal > Fmax condition

%number of states and inputs in dynamic model
nstates = 4;
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
Ey = 0.7;
Shy = 0;
Svy = 0;
g = 9.806;
Fmax = 0.7 * m * g;

%% Time Discretization
%Tfinal should be defined as a function speed, higher Tfinal for higher speed

%Sample constant time discretization
dt=0.01;
Tfinal = 2;
%time span
T = 0 : dt : Tfinal;

%% load reference trajectory
%What does first entry correspond to?
%How to define U, first column should be theta
%Y ref should correspond to centerline trajectory at initial index to final index for the time horizon

U_ref = interp1(0:0.01:Tfinal,[U,U(:,end)]',T)'; 
Y_ref = interp1(0:0.01:Tfinal,Y,T)'

%% Discrete-time A and B matrices
%these are the system linearized in discrete time about the reference
%trajectory i.e. x(i+1)=A_i*x_i+B_i*u_i

%Change to A, B from dynamics
A=@(i) eye(3)+dt*[0 0 -U_ref(1,i)*sin(Y_ref(3,i))-(U_ref(1,i)*b/L)*cos(Y_ref(3,i))*tan(U_ref(2,i));...
                     0 0  U_ref(1,i)*cos(Y_ref(3,i))-(U_ref(1,i)*b/L)*sin(Y_ref(3,i))*tan(U_ref(2,i));...
                      0 0                       0                    ];
B=@(i) dt*[cos(Y_ref(3,i))-b/L*sin(Y_ref(3,i))*tan(U_ref(2,i)), -(U_ref(1,i)*b*sin(Y_ref(3,i)))/(L*cos(U_ref(2,i))^2);...
           sin(Y_ref(3,i))+b/L*cos(Y_ref(3,i))*tan(U_ref(2,i)), (U_ref(1,i)*b*cos(Y_ref(3,i)))/(L*cos(U_ref(2,i))^2);...
           1/L*tan(U_ref(2,i)),   

%% Number of decision variables for colocation method
%11 timesteps for 3 states, 10 timesteps for 2 inputs

npred=10;
Ndec=(npred+1)*nstates+ninputs*npred;
%decision variable will be z=[x_1...x_11;u_1...u_10] (x refers to state
%vector, u refers to input vector)

%% Test function at index 1 to construct Aeq Beq (equality constraints
%enforce x(i+1)=A_i*x_i+B_i*u_i
eY0=[0 ; 0 ; 0];
[Aeq_test1,beq_test1]=eq_cons(1,A,B,eY0,npred,nstates,ninputs);

%% Test function at index 1 to generate limits on inputs
[Lb_test1,Ub_test1]=bound_cons(1,U_ref,npred,input_range,nstates,ninputs);


%% 4.5 simulate controller working

%final trajectory
Y=NaN(3,length(T));

%applied inputs
U=NaN(2,length(T));

%input from QP
u_mpc=NaN(2,length(T));

%error in states (actual-reference)
eY=NaN(3,length(T));

%set random initial condition

Y(:,1)=eY0+Y_ref(:,1);

for i=1:length(T)-1
    %shorten prediction horizon if we are at the end of trajectory
    npred_i=min([npred,length(T)-i]);
    
    %calculate error
    eY(:,i)=Y(:,i)-Y_ref(:,i);

    %generate equality constraints
    [Aeq,beq]=eq_cons(i,A,B,eY(:,i),npred_i,nstates,ninputs);
    
    %generate boundary constraints
    [Lb,Ub]=bound_cons(i,U_ref,npred_i,input_range,nstates,ninputs);
    
    %cost for states
    Q=[1,1,0.5];
    
    %cost for inputs
    R=[0.1,0.01];
    
    H=diag([repmat(Q,[1,npred_i+1]),repmat(R,[1,npred_i])]);
    
    f=zeros(nstates*(npred_i+1)+ninputs*npred_i,1);
    
    [x,fval] = quadprog(H,f,[],[],Aeq,beq,Lb,Ub);
    
    %get linearized input
    u_mpc(:,i)=x(nstates*(npred_i+1)+1:nstates*(npred_i+1)+ninputs);
    
    %get input
    U(:,i)=u_mpc(:,i)+U_ref(:,i);
    
    
    %simulate model
    [~,ztemp]=ode45(@(t,z)kinematic_bike_dynamics(t,z,U(:,i),0,b,L),[0 dt],Y(:,i));
    
    %store final state
    Y(:,i+1)=ztemp(end,:)';
end


%% Functions 

function [Aeq,beq] = eq_cons(initial_idx,A,B,x_initial,npred,nstates,ninputs)
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

function [Lb,Ub] = bound_cons(initial_idx,U_ref,npred,input_range,nstates,ninputs)
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

function dzdt=kinematic_bike_dynamics(t,x,T,U)
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
delta_f=interp1(T,U(:,1),t,'previous','extrap');
F_x=interp1(T,U(:,2),t,'previous','extrap');

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


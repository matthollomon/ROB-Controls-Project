%% Generate manual inputs for left and right trajectories


%% Load data

TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
thetaCline = TestTrack.TestTrack.theta;

dt = 0.01;

%% Create manual drive input

Uright = [
    repmat([-0.03, 2000],200,1);
    repmat([0, 2000],200,1);
    repmat([0.005, 1500],300,1);
    repmat([-0.005, -1500],200,1);
    repmat([-0.01, -2500],150,1);
    repmat([-0.045, 00],300,1);
    repmat([0.00, -2500],200,1);
    repmat([-0.02,1500],400,1);
    repmat([-0.04, 1500],100,1);
    repmat([-0.05, 1000],100,1);
    repmat([ 0.0025, 800],600,1);
    repmat([-0.005, -500],200,1);
    repmat([0.015, -1500],150,1);
    repmat([0.032, -1000],400,1);
    repmat([-0.02, -500],300,1);
    repmat([-0.015, 00],300,1);
    repmat([-0.08, 500],200,1);
    repmat([-0.01, 00],300,1);
    repmat([-0.0, 0],250,1);
    repmat([0.055, 00],300,1);
    repmat([0.045, 00],300,1);
    repmat([0.02, 100],300,1);
    repmat([0, 100],400,1);
    repmat([-0.04, 800],500,1);
    repmat([-0.0059, -700],300,1);
    repmat([-0.025, 00],400,1); 
    repmat([-0.045, 100],400,1);
    repmat([-0.005, 100],500,1);
    repmat([0.0095, 100],300,1);
    repmat([-0.015, 100],250,1);
    repmat([0.095, 100],300,1);
    repmat([0.014, 1000],600,1);
    repmat([-0.01, -1500],200,1);
    repmat([-0.05, -400],200,1);
    repmat([-0.057, -100],500,1);
    repmat([-0.02, 100],250,1);
    repmat([-0.01, -500],150,1);
    repmat([0.49, -100],150,1);
    repmat([0.038, 500],250,1);
    repmat([-0.0, 1000],420,1);
    repmat([-0.0007, 1100],580,1);
    repmat([0.45, -2100],150,1);
    repmat([0.05, 1000],100,1);
    repmat([-0.015, 1500],200,1);
    repmat([0.012, 500],200,1);
    repmat([-0.004, 100],300,1);
    repmat([0.0, 100],300,1);
    repmat([0.001,100],300,1);
    repmat([0.001, 100],300,1);
    repmat([0.04, 100],150,1);
   ];


Uleft = [
    repmat([0.01, 100],50,1);
    repmat([-0.01, 200],200,1)
    repmat([-0.0, 300],550,1);
    repmat([-0.01, 300],100,1);
    repmat([0.0, 400],550,1);
    repmat([-0.01, 300],500,1);
    repmat([-0.02, 200],50,1);
    repmat([-0.03, 200],100,1);
    repmat([-0.03, 300],400,1);
    repmat([-0.02, 200],300,1);
    repmat([-0.01, 200],400,1);
    repmat([0.0, 300],700,1);
    repmat([0.01, 200],50,1);
    repmat([0.02, 100],200,1);
    repmat([0.03, 0],150,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([-0.01, 0],150,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, -100],50,1);
    repmat([-0.02, -200],50,1);
    repmat([-0.01, -300],50,1);
    repmat([0.0, -200],50,1);
    repmat([0.01, -100],50,1);
    repmat([0.02, -200],50,1);
    repmat([0.03, -300],50,1);
    repmat([0.04, -200],50,1);
    repmat([0.05, 0],200,1);
    repmat([0.04, 0],100,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],50,1);
    repmat([-0.03, 0],100,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],300,1);
    repmat([-0.02, 0],600,1);
    repmat([-0.03, 0],350,1);
    repmat([-0.02, 0],150,1);
    repmat([-0.01, -100],350,1);
    repmat([0.0, 0],350,1);
    repmat([0.01, -100],50,1);
    repmat([0.02, -100],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, -100],50,1);
    repmat([0.06, -100],50,1);
    repmat([0.07, -100],50,1);
    repmat([0.08, -100],50,1);
    repmat([0.08, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.0, 0],350,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],200,1);
    repmat([0.01, 0],50,1);
    repmat([0.00, 0],150,1);
    repmat([-0.01, 0],50,1);
    repmat([-0.02, 0],150,1);
    repmat([-0.03, 100],200,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.06, 100],50,1);
    repmat([-0.07, 0],200,1);
    repmat([-0.06, 0],450,1);
    repmat([-0.05, 0],50,1);
    repmat([-0.04, 0],50,1);
    repmat([-0.03, 0],100,1);
    repmat([-0.02, 0],100,1);
    repmat([-0.01, 0],50,1);
    repmat([0.0, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.08, 0],50,1);
    repmat([0.09, 0],200,1);
    repmat([0.08, 0],50,1);
    repmat([0.07, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 100],50,1);
    repmat([0.00, 200],500,1);
    repmat([-0.01, 300],300,1);
    repmat([0.00, 400],700,1);
    repmat([0.01, 400],100,1);
    repmat([0.02, 400],50,1);
    repmat([0.03, 400],300,1);
    repmat([0.02, 400],200,1);
    repmat([0.01, 500],50,1);
    repmat([0.0, 600],1350,1);
    repmat([0.01, 700],300,1);
];

timeLeft = size(Uleft,1) * 0.01;
timeRight = size(Uright,1) * 0.01;
   
%% Save results

%U_ref = U;
% save('ROB535_ControlProject_part1_Team19','U')

%% Simulate

x0 = [287 , 5 , -176 , 0 , 2 , 0];
endpoint = [1470 , 810];

[Yleft,T] = forwardIntegrateControlInput(Uleft,x0);
[Yright,T] = forwardIntegrateControlInput(Uright,x0);

save('U_right','Uright')
save('Y_left','Yleft')

%% Plotting

figure(2)
yyaxis left
plot(T,U_ref(:,1),'b')
ylabel('Steeting Angle')
hold on
yyaxis right
plot(T,U_ref(:,2),'r')
ylabel('Acceleration')
xlabel('Time')
hold off

figure(3)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(cline(1,:),cline(2,:),'--k')
plot(Yleft(:,1),Yleft(:,3),'r')
plot(Yright(:,1),Yright(:,3),'b')
hold off



%% Generate manual inputs for left and right trajectories


%% Load data

TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
thetaCline = TestTrack.TestTrack.theta;

dt = 0.01;

%% Create manual drive input

U = [repmat([-0.05, 100],50,1);
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
    repmat([-0.03, 0],150,1);
    repmat([-0.02, 0],450,1);
    repmat([-0.03, 0],350,1);
    repmat([-0.02, 0],350,1);
    repmat([-0.01, 0],150,1);
    repmat([0, 0],450,1);
    repmat([0.01, 0],100,1);
    repmat([0, 0],1100,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.02, 0],200,1);
    repmat([0.03, 0],200,1);
    repmat([0.04, 0],250,1);
    repmat([0.03, 0],100,1);
    repmat([0.02, 0],50,1);
    repmat([0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([-0.01, 0],50,1);
    repmat([0.00, 0],50,1);
    repmat([0.00, 0],150,1);
    repmat([-.01, 0],250,1);
    repmat([-.02, 0],50,1);
    repmat([-.03, 0],50,1);
    repmat([-.04, 0],50,1);
    repmat([-.05, 0],100,1);
    repmat([-.04, 0],550,1);
    repmat([-.05, 0],100,1);
    repmat([-.04, 0],250,1);
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
    repmat([0.04, 0],300,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.02, 0],450,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.03, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.04, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.05, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.06, 0],50,1);
    repmat([0.07, 0],250,1);
    repmat([0.06, 0],100,1);
    repmat([0.05, 0],100,1);
    repmat([0.04, 0],600,1);
    repmat([0.05, 0],300,1);
    repmat([0.04, 0],150,1);
    repmat([0.03, 0],150,1);
    repmat([0.02, 0],150,1);
    repmat([0.01, 0],150,1);
    repmat([0.0, 0],900,1);
    
    ];

time = size(U,1) * 0.01;
   
%% Save results

%U_ref = U;
% save('ROB535_ControlProject_part1_Team19','U')

%% Simulate

x0 = [287 , 5 , -176 , 0 , 2 , 0];
endpoint = [1470 , 810];

[Y,T] = forwardIntegrateControlInput(U,x0);
Y_ref = Y;

%% Plotting

% figure(2)
% yyaxis left
% plot(T,U_ref(:,1),'b')
% ylabel('Steeting Angle')
% hold on
% yyaxis right
% plot(T,U_ref(:,2),'r')
% ylabel('Acceleration')
% xlabel('Time')
% hold off

figure(3)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(cline(1,:),cline(2,:),'--k')
plot(Y(:,1),Y(:,3),'r')
xlim([ 300 800])
ylim([ 100 400])
hold off



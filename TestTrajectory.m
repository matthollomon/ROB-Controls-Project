clear all
clc

%Generate sample reference input

TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
thetaCline = TestTrack.TestTrack.theta;

% for i = 1 : length(thetaCline) - 1
%     deltaTheta(i) = thetaCline(i + 1) - thetaCline(i);
% end
% 
% interp_size = 100;
% cline = [interp(cline(1,:),interp_size) ; interp(cline(2,:),interp_size)];
% deltaTheta = [interp(deltaTheta(:),interp_size)];

% figure(1)
% plot(cline(1,:) , cline(2,:) , 'k')
% hold on
% plot(1:length(thetaCline) , thetaCline , 'r')
% plot(1:length(deltaTheta) , deltaTheta , 'b')

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
    repmat([-0.06 , 0],50,1);
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
    repmat([0.0 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.02 , 0],50,1);
    repmat([0.1 , 0],50,1);
    repmat([0.1 , 0],50,1);
    repmat([0.20 , 0],50,1);
    repmat([0.05 , 0],50,1);
    repmat([0.03 , 0],50,1);
    repmat([0.02 , 0],50,1);
    repmat([0.01 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
    repmat([0.0 , 0],50,1);
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
    repmat([0.02 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.03 , 0],100,1);
    repmat([0.02 , 0],100,1);
    repmat([0.02, 0],100,1);
    repmat([0.01 , 0],100,1);
    repmat([0.01 , 0],100,1);
    repmat([0.012 , 0],200,1);
    repmat([0 , 5000],850,1);
    
    ];
    
time = size(U,1) * 0.01;

U_ref = U;
save('ROB535_ControlProject_part1_Team19','U')

% deltafSmooth = smooth(U(:,1));
% accSmooth = smooth(U(:,2)); 
% U_ref = [deltafSmooth , accSmooth];

x0 = [287 , 5 , -176 , 0 , 2 , 0];
endpoint = [1470 , 810];

[Y,T] = forwardIntegrateControlInput(U_ref,x0);
Y_ref = Y;



figure(2)
plot(bl(1,:),bl(2,:),'k')
hold on
plot(br(1,:),br(2,:),'k')
plot(cline(1,:),cline(2,:),'--k')
plot(Y(:,1),Y(:,3),'r')
%xlim([1100 1600])
%ylim([400 900])
hold off

figure(3)
yyaxis left
plot(T,U_ref(:,1),'r')
ylabel('Steeting Angle')
hold on
yyaxis right
plot(T,U_ref(:,2),'r')
ylabel('Acceleration')
xlabel('Time')
hold off


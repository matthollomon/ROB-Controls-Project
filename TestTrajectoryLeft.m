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

U_left = [repmat([0.1, 2400],100,1);
    repmat([-0.1, 2400],100,1);
    repmat([0.0, 2200],550,1);
    repmat([-0.85, 400],500,1);
    repmat([0, 0],150,1);
    repmat([-0.6, 0],90,1);
    repmat([0, 0],1300,1);
    repmat([0.05, 0],260,1);
    repmat([0, 0],400,1);
    repmat([-0.045, 0],400,1);
    repmat([0, 0],340,1);
    repmat([0.07, 0],350,1);
    repmat([0, 0],550,1);
    repmat([-0.04, 0],450,1);
    repmat([-0.01, 0],600,1);
    repmat([-0.037, 0],750,1);
    repmat([0, 400],875,1);
    repmat([0.08, 0],300,1);
    repmat([0, 0],330,1);
    repmat([0.03, 0],340,1);
    repmat([-0.042, 0],900,1);
    repmat([-0.06, 0],310,1);
    repmat([0, 0],150,1);
    repmat([0.13, 0],300,1);
    repmat([0, 500],1275,1);
    repmat([0.057, 0],200,1);
    repmat([0, 500],1600,1);
    repmat([0.03, 0],150,1);
]
   
time = size(U_left,1) * 0.01;

U_ref_left = U_left;
save('ROB535_ControlProject_part1_Team19','U_left')

% deltafSmooth = smooth(U(:,1));
% accSmooth = smooth(U(:,2)); 
% U_ref = [deltafSmooth , accSmooth];

x0 = [287 , 5 , -176 , 0 , 2 , 0];
endpoint = [1470 , 810];

[Y,T] = forwardIntegrateControlInput(U_ref_left,x0);
Y_ref = Y;

% x0 = Y(end,:);
% 
% while Y(1,end) < endpoint(1) && Y(3,end) < endpoint(2)
%     
%     [value , idx] = min(
%     deltaTheta = 
%     U = [U , [deltaTheta,0];
%     [Y,T] = forwardIntegrateControlInput(U,x0);
%     
%     x0 = Y(end,:);
% 
% end

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
plot(T,U_ref_left(:,1),'r')
ylabel('Steeting Angle')
hold on
yyaxis right
plot(T,U_ref_left(:,2),'r')
ylabel('Acceleration')
xlabel('Time')
hold off

TestTrack = load('TestTrack.mat')
bl = TestTrack.TestTrack.bl;
br = TestTrack.TestTrack.br;
cline = TestTrack.TestTrack.cline;
theta = TestTrack.TestTrack.theta;

d = 5;

ubX = cline(1,:) - d * cos(pi / 2 - theta);
ubY = cline(2,:) + d * sin(pi / 2 - theta);

lbX = cline(1,:) + d * cos(pi / 2 - theta);
lbY = cline(2,:) - d * sin(pi / 2 - theta);

figure(1)
plot(cline(1,:) , cline(2,:) , 'b')
hold on
plot(bl(1,:),bl(2,:),'k')
plot(br(1,:),br(2,:),'k')
plot(lbX,lbY , 'r')
plot(ubX,ubY,'r')
hold off
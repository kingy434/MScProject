%%%Load the data
Fs = 200;
Y1 = phys(1).RW.accel;
Y1l1 = labels(2).premed_tremorfinal;
Y1l2 = labels(2).postmed_tremorfinal;
Ylabel1 = [Y1l1; Y1l2];
premed_tremor_start1 = labels(2).premed_tremorstart.*Fs;
premed_tremor_end1 = labels(2).premed_tremorend.*Fs;

postmed_tremor_start1 = labels(2).postmed_tremorstart.*Fs;
postmed_tremor_end1 = labels(2).postmed_tremorend.*Fs;

tremor_type_concat1 = [];
tremor_data_concat1 = [];
num_proto = length(labels(2).table_tremorproto.Start);

Y_proto1 = zeros(1,length(Y1));
for t = 1:num_proto
    tremor_type_start1(t) = labels(2).table_tremorproto.Start(t);
    tremor_type_end1(t) = labels(2).table_tremorproto.End(t);
    tremor_type1(t) = labels(2).table_tremorproto.Code(t);
    tremor_ind1 = (phys(1).RW.accel(:, 1) > tremor_type_start1(t) & phys(1).RW.accel(:, 1)< tremor_type_end1(t));
    Y_proto1(tremor_ind1) = double(string(tremor_type1(t)));
    tremor_data_per_type1{t} = phys(1).RW.accel(tremor_ind1, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type1{t}),1).*double(string(tremor_type1(t)));
    tremor_type_concat1 = [tremor_type_concat1; tremor_type_concat];
    tremor_data_concat1 = [tremor_data_concat1;phys(1).RW.accel(tremor_ind1, 2:4)];
end
Y_proto1 = [Y_proto1(premed_tremor_start1:premed_tremor_end1), Y_proto1(postmed_tremor_start1:postmed_tremor_end1)];
Y_tremor_premed1 = phys(1).RW.accel(premed_tremor_start1:premed_tremor_end1, 2:4);
Y_tremor_postmed1 = phys(1).RW.accel(postmed_tremor_start1:postmed_tremor_end1, 2:4);
Y_tremor1 = [Y_tremor_premed1.*9.8; Y_tremor_postmed1.*9.8];
%%
Y2 = phys(6).LW.accel;
Y2l1 = labels(6).premed_tremorfinal;
Y2l2 = labels(6).postmed_tremorfinal;
Ylabel2 = [Y2l1; Y2l2];
premed_tremor_start2 = labels(6).premed_tremorstart.*Fs;
premed_tremor_end2 = labels(6).premed_tremorend.*Fs;
postmed_tremor_start2 = labels(6).postmed_tremorstart.*Fs;
postmed_tremor_end2 = labels(6).postmed_tremorend.*Fs;

tremor_type_concat2 = [];
tremor_data_concat2 = [];
num_proto = length(labels(6).table_tremorproto.Start);

Y_proto2 = zeros(1,length(Y2));

for t = 1:num_proto
    tremor_type_start2(t) = labels(6).table_tremorproto.Start(t);
    tremor_type_end2(t) = labels(6).table_tremorproto.End(t);
    tremor_type2(t) = labels(6).table_tremorproto.Code(t);
    tremor_ind2 = (phys(6).LW.accel(:, 1) > tremor_type_start2(t) & phys(6).LW.accel(:, 1)< tremor_type_end2(t));
    Y_proto2(tremor_ind2) = double(string(tremor_type2(t)));
    tremor_data_per_type2{t} = phys(6).LW.accel(tremor_ind2, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type2{t}),1).*double(string(tremor_type2(t)));
    tremor_type_concat2 = [tremor_type_concat2; tremor_type_concat];
    tremor_data_concat2 = [tremor_data_concat2;phys(6).LW.accel(tremor_ind2, 2:4)];
end
Y_proto2 = [Y_proto2(premed_tremor_start2:premed_tremor_end2), Y_proto2(postmed_tremor_start2:postmed_tremor_end2)];
Y_tremor_premed2 = phys(6).LW.accel(premed_tremor_start2:premed_tremor_end2, 2:4);
Y_tremor_postmed2 = phys(6).LW.accel(postmed_tremor_start2:postmed_tremor_end2, 2:4);
Y_tremor2 = [Y_tremor_premed2.*9.8; Y_tremor_postmed2.*9.8];
%%
Y3 = phys(8).RW.accel;
Y3l1 = labels(7).premed_tremorfinal;
Y3l2 = labels(7).postmed_tremorfinal;
Ylabel3 = [Y3l1; Y3l2];
premed_tremor_start3 = labels(7).premed_tremorstart.*Fs;
premed_tremor_end3 = labels(7).premed_tremorend.*Fs;
postmed_tremor_start3 = labels(7).postmed_tremorstart.*Fs;
postmed_tremor_end3 = labels(7).postmed_tremorend.*Fs;

tremor_type_concat3 = [];
tremor_data_concat3 = [];
num_proto = length(labels(7).table_tremorproto.Start);

Y_proto3 = zeros(1,length(Y3));

for t = 1:num_proto
    tremor_type_start3(t) = labels(7).table_tremorproto.Start(t);
    tremor_type_end3(t) = labels(7).table_tremorproto.End(t);
    tremor_type3(t) = labels(7).table_tremorproto.Code(t);
    tremor_ind3 = (phys(8).RW.accel(:, 1) > tremor_type_start3(t) & phys(8).RW.accel(:, 1)< tremor_type_end3(t));
    Y_proto3(tremor_ind3) = double(string(tremor_type3(t)));
    tremor_data_per_type3{t} = phys(8).RW.accel(tremor_ind3, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type3{t}),1).*double(string(tremor_type3(t)));
    tremor_type_concat3 = [tremor_type_concat3; tremor_type_concat];
    tremor_data_concat3 = [tremor_data_concat3;phys(8).RW.accel(tremor_ind3, 2:4)];
end
Y_proto3 = [Y_proto3(premed_tremor_start3:premed_tremor_end3), Y_proto3(postmed_tremor_start3:postmed_tremor_end3)];
Y_tremor_premed3 = phys(8).RW.accel(premed_tremor_start3:premed_tremor_end3, 2:4);
Y_tremor_postmed3 = phys(8).RW.accel(postmed_tremor_start3:postmed_tremor_end3, 2:4);
Y_tremor3 = [Y_tremor_premed3.*9.8; Y_tremor_postmed3.*9.8];
%%
Y4 = phys(25).LW.accel;
Y4l1 = labels(18).premed_tremorfinal;
Y4l2 = labels(18).postmed_tremorfinal;
Ylabel4 = [Y4l1; Y4l2];
Ylabel4 = Ylabel4(1:end-1);
premed_tremor_start4 = labels(18).premed_tremorstart.*Fs;
premed_tremor_end4 = labels(18).premed_tremorend.*Fs;
postmed_tremor_start4 = labels(18).postmed_tremorstart.*Fs;
postmed_tremor_end4 = labels(18).postmed_tremorend.*Fs;

tremor_type_concat4 = [];
tremor_data_concat4 = [];
num_proto = length(labels(18).table_tremorproto.Start);

Y_proto4 = zeros(1,length(Y4));
for t = 1:num_proto
    tremor_type_start4(t) = labels(18).table_tremorproto.Start(t);
    tremor_type_end4(t) = labels(18).table_tremorproto.End(t);
    tremor_type4(t) = labels(18).table_tremorproto.Code(t);
    tremor_ind4 = (phys(25).LW.accel(:, 1) > tremor_type_start4(t) & phys(25).LW.accel(:, 1)< tremor_type_end4(t));
    Y_proto4(tremor_ind4) = double(string(tremor_type4(t)));
    tremor_data_per_type4{t} = phys(25).LW.accel(tremor_ind4, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type4{t}),1).*double(string(tremor_type4(t)));
    tremor_type_concat4 = [tremor_type_concat4; tremor_type_concat];
    tremor_data_concat4 = [tremor_data_concat4;phys(25).LW.accel(tremor_ind4, 2:4)];
end
Y_proto4 = [Y_proto4(premed_tremor_start4:premed_tremor_end4), Y_proto4(postmed_tremor_start4:postmed_tremor_end4)];
Y_tremor_premed4 = phys(25).LW.accel(premed_tremor_start4:premed_tremor_end4, 2:4);
Y_tremor_postmed4 = phys(25).LW.accel(postmed_tremor_start4:postmed_tremor_end4, 2:4);
Y_tremor4 = [Y_tremor_premed4.*9.8; Y_tremor_postmed4.*9.8];
%%
Y5 = phys(2).LW.accel;
Y5l1 = labels(19).premed_tremorfinal;
Y5l2 = labels(19).postmed_tremorfinal;
Ylabel5 = [Y5l1; Y5l2];
premed_tremor_start5 = labels(19).premed_tremorstart.*Fs;
premed_tremor_end5 = labels(19).premed_tremorend.*Fs;
postmed_tremor_start5 = labels(19).postmed_tremorstart.*Fs;
postmed_tremor_end5 = labels(19).postmed_tremorend.*Fs;

tremor_type_concat5 = [];
tremor_data_concat5 = [];
num_proto = length(labels(19).table_tremorproto.Start);

Y_proto5 = zeros(1,length(Y5));
for t = 1:num_proto
    tremor_type_start5(t) = labels(19).table_tremorproto.Start(t);
    tremor_type_end5(t) = labels(19).table_tremorproto.End(t);
    tremor_type5(t) = labels(19).table_tremorproto.Code(t);
    tremor_ind5 = (phys(2).LW.accel(:, 1) > tremor_type_start5(t) & phys(2).LW.accel(:, 1)< tremor_type_end5(t));
    Y_proto5(tremor_ind5) = double(string(tremor_type5(t)));
    tremor_data_per_type5{t} = phys(2).LW.accel(tremor_ind5, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type5{t}),1).*double(string(tremor_type5(t)));
    tremor_type_concat5 = [tremor_type_concat5; tremor_type_concat];
    tremor_data_concat5 = [tremor_data_concat5;phys(2).LW.accel(tremor_ind5, 2:4)];
end
Y_proto5 = [Y_proto5(premed_tremor_start5:premed_tremor_end5), Y_proto5(postmed_tremor_start5:postmed_tremor_end5)];
Y_tremor_premed5 = phys(2).LW.accel(premed_tremor_start5:premed_tremor_end5, 2:4);
Y_tremor_postmed5 = phys(2).LW.accel(postmed_tremor_start5:postmed_tremor_end5, 2:4);
Y_tremor5 = [Y_tremor_premed5.*9.8; Y_tremor_postmed5.*9.8];
%%
Y6 = phys(7).RW.accel;
Y6l1 = labels(20).premed_tremorfinal;
Y6l2 = labels(20).postmed_tremorfinal;
Ylabel6 = [Y6l1; Y6l2];
premed_tremor_start6 = labels(20).premed_tremorstart.*Fs;
premed_tremor_end6 = labels(20).premed_tremorend.*Fs;
postmed_tremor_start6 = labels(20).postmed_tremorstart.*Fs;
postmed_tremor_end6 = labels(20).postmed_tremorend.*Fs;

tremor_type_concat6 = [];
tremor_data_concat6 = [];
num_proto = length(labels(20).table_tremorproto.Start);

Y_proto6 = zeros(1,length(Y6));
for t = 1:num_proto
    tremor_type_start6(t) = labels(20).table_tremorproto.Start(t);
    tremor_type_end6(t) = labels(20).table_tremorproto.End(t);
    tremor_type6(t) = labels(20).table_tremorproto.Code(t);
    tremor_ind6 = (phys(7).RW.accel(:, 1) > tremor_type_start6(t) & phys(7).RW.accel(:, 1)< tremor_type_end6(t));
    Y_proto6(tremor_ind6) = double(string(tremor_type6(t)));
    tremor_data_per_type6{t} = phys(7).RW.accel(tremor_ind6, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type6{t}),1).*double(string(tremor_type6(t)));
    tremor_type_concat6 = [tremor_type_concat6; tremor_type_concat];
    tremor_data_concat6 = [tremor_data_concat6;phys(7).RW.accel(tremor_ind6, 2:4)];
end
Y_proto6 = [Y_proto6(premed_tremor_start6:premed_tremor_end6), Y_proto6(postmed_tremor_start6:postmed_tremor_end6)];
Y_tremor_premed6 = phys(7).RW.accel(premed_tremor_start6:premed_tremor_end6, 2:4);
Y_tremor_postmed6 = phys(7).RW.accel(postmed_tremor_start6:postmed_tremor_end6, 2:4);
Y_tremor6 = [Y_tremor_premed6.*9.8; Y_tremor_postmed6.*9.8];
%%
Y7 = phys(9).LW.accel;
Y7l1 = labels(21).premed_tremorfinal;
Y7l2 = labels(21).postmed_tremorfinal;
Ylabel7 = [Y7l1; Y7l2];
premed_tremor_start7 = labels(21).premed_tremorstart.*Fs;
premed_tremor_end7 = labels(21).premed_tremorend.*Fs;
postmed_tremor_start7 = labels(21).postmed_tremorstart.*Fs;
postmed_tremor_end7 = labels(21).postmed_tremorend.*Fs;

tremor_type_concat7 = [];
tremor_data_concat7 = [];
num_proto = length(labels(21).table_tremorproto.Start);

Y_proto7 = zeros(1,length(Y7));
for t = 1:num_proto
    tremor_type_start7(t) = labels(21).table_tremorproto.Start(t);
    tremor_type_end7(t) = labels(21).table_tremorproto.End(t);
    tremor_type7(t) = labels(21).table_tremorproto.Code(t);
    tremor_ind7 = (phys(9).LW.accel(:, 1) > tremor_type_start7(t) & phys(9).LW.accel(:, 1)< tremor_type_end7(t));
    Y_proto7(tremor_ind7) = double(string(tremor_type7(t)));
    tremor_data_per_type7{t} = phys(9).LW.accel(tremor_ind7, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type7{t}),1).*double(string(tremor_type7(t)));
    tremor_type_concat7 = [tremor_type_concat7; tremor_type_concat];
    tremor_data_concat7 = [tremor_data_concat7;phys(9).LW.accel(tremor_ind7, 2:4)];
end
Y_proto7 = [Y_proto7(premed_tremor_start7:premed_tremor_end7), Y_proto7(postmed_tremor_start7:postmed_tremor_end7)];
Y_tremor_premed7 = phys(9).LW.accel(premed_tremor_start7:premed_tremor_end7, 2:4);
Y_tremor_postmed7 = phys(9).LW.accel(postmed_tremor_start7:postmed_tremor_end7, 2:4);
Y_tremor7 = [Y_tremor_premed7.*9.8; Y_tremor_postmed7.*9.8];
%%
Y8 = phys(11).RW.accel;
Y8l1 = labels(22).premed_tremorfinal;
Y8l2 = labels(22).postmed_tremorfinal;
Ylabel8 = [Y8l1; Y8l2];
premed_tremor_start8 = labels(22).premed_tremorstart.*Fs;
premed_tremor_end8 = labels(22).premed_tremorend.*Fs;
postmed_tremor_start8 = labels(22).postmed_tremorstart.*Fs;
postmed_tremor_end8 = labels(22).postmed_tremorend.*Fs;

tremor_type_concat8 = [];
tremor_data_concat8 = [];
num_proto = length(labels(22).table_tremorproto.Start);

Y_proto8 = zeros(1,length(Y8));
for t = 1:num_proto
    tremor_type_start8(t) = labels(22).table_tremorproto.Start(t);
    tremor_type_end8(t) = labels(22).table_tremorproto.End(t);
    tremor_type8(t) = labels(22).table_tremorproto.Code(t);
    tremor_ind8 = (phys(11).RW.accel(:, 1) > tremor_type_start8(t) & phys(11).RW.accel(:, 1)< tremor_type_end8(t));
    Y_proto8(tremor_ind8) = double(string(tremor_type8(t)));
    tremor_data_per_type8{t} = phys(11).RW.accel(tremor_ind8, 2:4);
    tremor_type_concat = ones(length(tremor_data_per_type8{t}),1).*double(string(tremor_type8(t)));
    tremor_type_concat8 = [tremor_type_concat8; tremor_type_concat];
    tremor_data_concat8 = [tremor_data_concat8;phys(11).RW.accel(tremor_ind8, 2:4)];
end
Y_proto8 = [Y_proto8(premed_tremor_start8:premed_tremor_end8), Y_proto8(postmed_tremor_start8:postmed_tremor_end8)];
Y_tremor_premed8 = phys(11).RW.accel(premed_tremor_start8:premed_tremor_end8, 2:4);
Y_tremor_postmed8 = phys(11).RW.accel(postmed_tremor_start8:postmed_tremor_end8, 2:4);
Y_tremor8 = [Y_tremor_premed8.*9.8; Y_tremor_postmed8.*9.8];
%%
Ind1 = (Ylabel1 == '1' | Ylabel1 == '2');
Ind2 = (Ylabel2 == '1' | Ylabel2 == '2');
Ind3 = (Ylabel3 == '1' | Ylabel3 == '2');
Ind4 = (Ylabel4 == '1' | Ylabel4 == '2');
Ind5 = (Ylabel5 == '1' | Ylabel5 == '2');
Ind6 = (Ylabel6 == '1' | Ylabel6 == '2');
Ind7 = (Ylabel7 == '1' | Ylabel7 == '2');
Ind8 = (Ylabel8 == '1' | Ylabel8 == '2');

figure;
subplot(4,2,1);plot(Y_tremor1(Ind1,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,2);plot(Y_tremor2(Ind2,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,3);plot(Y_tremor3(Ind3,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,4);plot(Y_tremor4(Ind4,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,5);plot(Y_tremor5(Ind5,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,6);plot(Y_tremor6(Ind6,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,7);plot(Y_tremor7(Ind7,:));
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

subplot(4,2,8);plot(Y_tremor8(Ind8,:));
xlabel("time (minutes)",'FontSize',16);
xticks([0.5 1 1.5 2].*30*Fs)
xticklabels({'0.25','0.5','0.75','1'})
ylabel("acceleration in m/s^2",'FontSize',16);
yticks([-20 -10 0 10 20]);
grid on
axis tight

figure;
subplot(4,2,1);plot(tremor_data_concat1);
hold on;
plot(tremor_type_concat1)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,2);plot(tremor_data_concat2);hold on;plot(tremor_type_concat2)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,3);plot(tremor_data_concat3);hold on;plot(tremor_type_concat3)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,4);plot(tremor_data_concat4);hold on;plot(tremor_type_concat4)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,5);plot(tremor_data_concat5);hold on;plot(tremor_type_concat5)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,6);plot(tremor_data_concat6);hold on;plot(tremor_type_concat6)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,7);plot(tremor_data_concat7);hold on;plot(tremor_type_concat7)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight
subplot(4,2,8);plot(tremor_data_concat8);hold on;plot(tremor_type_concat8)
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30*Fs)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
yticks([-20 -10 0 10 20]);
grid on
axis tight

Yx1 = Y_tremor1(Ind1, 1);
interval_count1 = floor(length(Yx1)/(Fs*30));% 30 seconds of data
interval_length1 = Fs*30;

for i = 1:interval_count1
    YYx1(i,:) = Yx1(((i-1)*interval_length1+1):i*interval_length1);
end

for k = 1:interval_count1
    [pxx_x1(k,:),f1] = pwelch(YYx1(k,:), 6000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x1,2)-1)));
%    select displayed section 
pxx_select_x1  = pxx_x1(:,1:M);
% 2) transpose matrix
pxx_reshape_x1 = transpose(pxx_select_x1);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx1,2)/Fs)*[0:size(YYx1,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x1      = 10*log10(pxx_reshape_x1);

figure;
subplot(4,2,1)
P1 = [pxx_dB_x1 pxx_dB_x1(:,end); pxx_dB_x1(end,:) pxx_dB_x1(end,end)];
pcolor(t,f(3:end),P1(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x1))-60 max(max(pxx_dB_x1))]);
colormap(hot);

xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
%%
Yx2 = Y_tremor2(Ind2, 1);
interval_count2 = floor(length(Yx2)/(Fs*10));% 30 seconds of data
interval_length2 = Fs*10;

for i = 1:interval_count2
    YYx2(i,:) = Yx2(((i-1)*interval_length2+1):i*interval_length2);
end

for k = 1:interval_count2
    [pxx_x2(k,:),f2] = pwelch(YYx2(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x2,2)-1)));
%    select displayed section 
pxx_select_x2  = pxx_x2(:,1:M);
% 2) transpose matrix
pxx_reshape_x2 = transpose(pxx_select_x2);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx2,2)/Fs)*[0:size(YYx2,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x2      = 10*log10(pxx_reshape_x2);

subplot(4,2,2)
P2 = [pxx_dB_x2 pxx_dB_x2(:,end); pxx_dB_x2(end,:) pxx_dB_x2(end,end)];
pcolor(t,f(3:end),P2(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x2))-60 max(max(pxx_dB_x2))]);
colormap(hot);

xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx3 = Y_tremor3(Ind3, 1);
interval_count3 = floor(length(Yx3)/(Fs*10));% 30 seconds of data
interval_length3 = Fs*10;

for i = 1:interval_count3
    YYx3(i,:) = Yx3(((i-1)*interval_length3+1):i*interval_length3);
end

for k = 1:interval_count3
    [pxx_x3(k,:),f3] = pwelch(YYx3(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x3,2)-1)));
%    select displayed section 
pxx_select_x3  = pxx_x3(:,1:M);
% 2) transpose matrix
pxx_reshape_x3 = transpose(pxx_select_x3);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx3,2)/Fs)*[0:size(YYx3,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x3      = 10*log10(pxx_reshape_x3);
% 5) generate plot
subplot(4,2,3)
P3 = [pxx_dB_x3 pxx_dB_x3(:,end); pxx_dB_x3(end,:) pxx_dB_x3(end,end)];
pcolor(t,f(3:end),P3(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x3))-70 max(max(pxx_dB_x3))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx4 = Y_tremor4(Ind4, 1);
interval_count4 = floor(length(Yx4)/(Fs*10));% 30 seconds of data
interval_length4 = Fs*10;

for i = 1:interval_count4
    YYx4(i,:) = Yx4(((i-1)*interval_length4+1):i*interval_length4);
end

for k = 1:interval_count4
    [pxx_x4(k,:),f4] = pwelch(YYx4(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x4,2)-1)));
%    select displayed section 
pxx_select_x4  = pxx_x4(:,1:M);
% 2) transpose matrix
pxx_reshape_x4 = transpose(pxx_select_x4);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx4,2)/Fs)*[0:size(YYx4,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x4      = 10*log10(pxx_reshape_x4);

subplot(4,2,4)
P4 = [pxx_dB_x4 pxx_dB_x4(:,end); pxx_dB_x4(end,:) pxx_dB_x4(end,end)];
pcolor(t,f(3:end),P4(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x4))-60 max(max(pxx_dB_x4))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx5 = Y_tremor5(Ind5, 1);
interval_count5 = floor(length(Yx5)/(Fs*10));% 30 seconds of data
interval_length5 = Fs*10;

for i = 1:interval_count5
    YYx5(i,:) = Yx5(((i-1)*interval_length5+1):i*interval_length5);
end

for k = 1:interval_count5
    [pxx_x5(k,:),f5] = pwelch(YYx5(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x5,2)-1)));
%    select displayed section 
pxx_select_x5  = pxx_x5(:,1:M);
% 2) transpose matrix
pxx_reshape_x5 = transpose(pxx_select_x5);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx5,2)/Fs)*[0:size(YYx5,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x5      = 10*log10(pxx_reshape_x5);

subplot(4,2,5)
P5 = [pxx_dB_x5 pxx_dB_x5(:,end); pxx_dB_x5(end,:) pxx_dB_x5(end,end)];
pcolor(t,f(3:end),P5(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x5))-60 max(max(pxx_dB_x5))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx6 = Y_tremor6(Ind6, 1);
interval_count6 = floor(length(Yx6)/(Fs*10));% 30 seconds of data
interval_length6 = Fs*10;

for i = 1:interval_count6
    YYx6(i,:) = Yx6(((i-1)*interval_length6+1):i*interval_length6);
end

for k = 1:interval_count6
    [pxx_x6(k,:),f6] = pwelch(YYx6(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x6,2)-1)));
%    select displayed section 
pxx_select_x6  = pxx_x6(:,1:M);
% 2) transpose matrix
pxx_reshape_x6 = transpose(pxx_select_x6);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx6,2)/Fs)*[0:size(YYx6,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x6      = 10*log10(pxx_reshape_x6);

subplot(4,2,6)
P6 = [pxx_dB_x6 pxx_dB_x6(:,end); pxx_dB_x6(end,:) pxx_dB_x6(end,end)];
pcolor(t,f(3:end),P6(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x6))-60 max(max(pxx_dB_x6))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx7 = Y_tremor7(Ind7, 1);
interval_count7 = floor(length(Yx7)/(Fs*10));% 30 seconds of data
interval_length7 = Fs*10;

for i = 1:interval_count7
    YYx7(i,:) = Yx7(((i-1)*interval_length7+1):i*interval_length7);
end

for k = 1:interval_count7
    [pxx_x7(k,:),f7] = pwelch(YYx7(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x7,2)-1)));
%    select displayed section 
pxx_select_x7  = pxx_x7(:,1:M);
% 2) transpose matrix
pxx_reshape_x7 = transpose(pxx_select_x7);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx7,2)/Fs)*[0:size(YYx7,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x7      = 10*log10(pxx_reshape_x7);

subplot(4,2,7)
P7 = [pxx_dB_x7 pxx_dB_x7(:,end); pxx_dB_x7(end,:) pxx_dB_x7(end,end)];
pcolor(t,f(3:end),P7(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x7))-60 max(max(pxx_dB_x7))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx8 = Y_tremor8(Ind8, 1);
interval_count8 = floor(length(Yx8)/(Fs*10));% 30 seconds of data
interval_length8 = Fs*10;

for i = 1:interval_count8
    YYx8(i,:) = Yx8(((i-1)*interval_length8+1):i*interval_length8);
end

for k = 1:interval_count8
    [pxx_x8(k,:),f8] = pwelch(YYx8(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x8,2)-1)));
%    select displayed section 
pxx_select_x8  = pxx_x8(:,1:M);
% 2) transpose matrix
pxx_reshape_x8 = transpose(pxx_select_x8);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx8,2)/Fs)*[0:size(YYx8,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x8      = 10*log10(pxx_reshape_x8);
subplot(4,2,8)
% Note: extend by one row & column since pcolor does not show the last row/col
P8 = [pxx_dB_x8 pxx_dB_x8(:,end); pxx_dB_x8(end,:) pxx_dB_x8(end,end)];
pcolor(t,f(3:end),P8(3:end,:)); shading flat;
% 6) choose dynamic range
% assign e.g. 60dB below peak power to the first value in the colormap
%             and the peak power to the last value in the colormap
caxis([max(max(pxx_dB_x8))-60 max(max(pxx_dB_x8))]);
% 7) select colormap
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%% Spectrum of the different tremor prototypes
clear pxx_x1 pxx_x2 pxx_x3 pxx_x4 pxx_x5 pxx_x6 pxx_x7 pxx_x8 YYx1 YYx2 YYx3 YYx4 YYx5 YYx6 YYx7 YYx8
Yx1 = tremor_data_concat1;
interval_count1 = floor(length(Yx1)/(Fs*30));% 30 seconds of data
interval_length1 = Fs*30;

for i = 1:interval_count1
    YYx1(i,:) = Yx1(((i-1)*interval_length1+1):i*interval_length1);
end

for k = 1:interval_count1
    [pxx_x1(k,:),f1] = pwelch(YYx1(k,:), 6000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x1,2)-1)));
%    select displayed section 
pxx_select_x1  = pxx_x1(:,1:M);
% 2) transpose matrix
pxx_reshape_x1 = transpose(pxx_select_x1);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx1,2)/Fs)*[0:size(YYx1,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x1      = 10*log10(pxx_reshape_x1);

figure;
subplot(4,2,1)
P1 = [pxx_dB_x1 pxx_dB_x1(:,end); pxx_dB_x1(end,:) pxx_dB_x1(end,end)];
pcolor(t,f(3:end),P1(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x1))-60 max(max(pxx_dB_x1))]);
colormap(hot);

xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
%%
Yx2 = tremor_data_concat2;
interval_count2 = floor(length(Yx2)/(Fs*10));% 30 seconds of data
interval_length2 = Fs*10;

for i = 1:interval_count2
    YYx2(i,:) = Yx2(((i-1)*interval_length2+1):i*interval_length2);
end

for k = 1:interval_count2
    [pxx_x2(k,:),f2] = pwelch(YYx2(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x2,2)-1)));
%    select displayed section 
pxx_select_x2  = pxx_x2(:,1:M);
% 2) transpose matrix
pxx_reshape_x2 = transpose(pxx_select_x2);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx2,2)/Fs)*[0:size(YYx2,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x2      = 10*log10(pxx_reshape_x2);

subplot(4,2,2)
P2 = [pxx_dB_x2 pxx_dB_x2(:,end); pxx_dB_x2(end,:) pxx_dB_x2(end,end)];
pcolor(t,f(3:end),P2(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x2))-60 max(max(pxx_dB_x2))]);
colormap(hot);

xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx3 = tremor_data_concat3;
interval_count3 = floor(length(Yx3)/(Fs*10));% 30 seconds of data
interval_length3 = Fs*10;

for i = 1:interval_count3
    YYx3(i,:) = Yx3(((i-1)*interval_length3+1):i*interval_length3);
end

for k = 1:interval_count3
    [pxx_x3(k,:),f3] = pwelch(YYx3(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x3,2)-1)));
%    select displayed section 
pxx_select_x3  = pxx_x3(:,1:M);
% 2) transpose matrix
pxx_reshape_x3 = transpose(pxx_select_x3);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx3,2)/Fs)*[0:size(YYx3,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x3      = 10*log10(pxx_reshape_x3);
% 5) generate plot
subplot(4,2,3)
P3 = [pxx_dB_x3 pxx_dB_x3(:,end); pxx_dB_x3(end,:) pxx_dB_x3(end,end)];
pcolor(t,f(3:end),P3(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x3))-70 max(max(pxx_dB_x3))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx4 = tremor_data_concat4;
interval_count4 = floor(length(Yx4)/(Fs*10));% 30 seconds of data
interval_length4 = Fs*10;

for i = 1:interval_count4
    YYx4(i,:) = Yx4(((i-1)*interval_length4+1):i*interval_length4);
end

for k = 1:interval_count4
    [pxx_x4(k,:),f4] = pwelch(YYx4(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x4,2)-1)));
%    select displayed section 
pxx_select_x4  = pxx_x4(:,1:M);
% 2) transpose matrix
pxx_reshape_x4 = transpose(pxx_select_x4);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx4,2)/Fs)*[0:size(YYx4,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x4      = 10*log10(pxx_reshape_x4);

subplot(4,2,4)
P4 = [pxx_dB_x4 pxx_dB_x4(:,end); pxx_dB_x4(end,:) pxx_dB_x4(end,end)];
pcolor(t,f(3:end),P4(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x4))-60 max(max(pxx_dB_x4))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx5 = tremor_data_concat5;
interval_count5 = floor(length(Yx5)/(Fs*10));% 30 seconds of data
interval_length5 = Fs*10;

for i = 1:interval_count5
    YYx5(i,:) = Yx5(((i-1)*interval_length5+1):i*interval_length5);
end

for k = 1:interval_count5
    [pxx_x5(k,:),f5] = pwelch(YYx5(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x5,2)-1)));
%    select displayed section 
pxx_select_x5  = pxx_x5(:,1:M);
% 2) transpose matrix
pxx_reshape_x5 = transpose(pxx_select_x5);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx5,2)/Fs)*[0:size(YYx5,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x5      = 10*log10(pxx_reshape_x5);

subplot(4,2,5)
P5 = [pxx_dB_x5 pxx_dB_x5(:,end); pxx_dB_x5(end,:) pxx_dB_x5(end,end)];
pcolor(t,f(3:end),P5(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x5))-60 max(max(pxx_dB_x5))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx6 = tremor_data_concat6;
interval_count6 = floor(length(Yx6)/(Fs*10));% 30 seconds of data
interval_length6 = Fs*10;

for i = 1:interval_count6
    YYx6(i,:) = Yx6(((i-1)*interval_length6+1):i*interval_length6);
end

for k = 1:interval_count6
    [pxx_x6(k,:),f6] = pwelch(YYx6(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x6,2)-1)));
%    select displayed section 
pxx_select_x6  = pxx_x6(:,1:M);
% 2) transpose matrix
pxx_reshape_x6 = transpose(pxx_select_x6);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx6,2)/Fs)*[0:size(YYx6,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x6      = 10*log10(pxx_reshape_x6);

subplot(4,2,6)
P6 = [pxx_dB_x6 pxx_dB_x6(:,end); pxx_dB_x6(end,:) pxx_dB_x6(end,end)];
pcolor(t,f(3:end),P6(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x6))-60 max(max(pxx_dB_x6))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx7 = tremor_data_concat7;
interval_count7 = floor(length(Yx7)/(Fs*10));% 30 seconds of data
interval_length7 = Fs*10;

for i = 1:interval_count7
    YYx7(i,:) = Yx7(((i-1)*interval_length7+1):i*interval_length7);
end

for k = 1:interval_count7
    [pxx_x7(k,:),f7] = pwelch(YYx7(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x7,2)-1)));
%    select displayed section 
pxx_select_x7  = pxx_x7(:,1:M);
% 2) transpose matrix
pxx_reshape_x7 = transpose(pxx_select_x7);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx7,2)/Fs)*[0:size(YYx7,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x7      = 10*log10(pxx_reshape_x7);

subplot(4,2,7)
P7 = [pxx_dB_x7 pxx_dB_x7(:,end); pxx_dB_x7(end,:) pxx_dB_x7(end,end)];
pcolor(t,f(3:end),P7(3:end,:)); shading flat;
caxis([max(max(pxx_dB_x7))-60 max(max(pxx_dB_x7))]);
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight

%%
Yx8 = tremor_data_concat8;
interval_count8 = floor(length(Yx8)/(Fs*10));% 30 seconds of data
interval_length8 = Fs*10;

for i = 1:interval_count8
    YYx8(i,:) = Yx8(((i-1)*interval_length8+1):i*interval_length8);
end

for k = 1:interval_count8
    [pxx_x8(k,:),f8] = pwelch(YYx8(k,:), 2000 , [], [], Fs);
end
% 1) Compute maximum frequency index
Fmax = 15; % Hz
M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x8,2)-1)));
%    select displayed section 
pxx_select_x8  = pxx_x8(:,1:M);
% 2) transpose matrix
pxx_reshape_x8 = transpose(pxx_select_x8);
% 3) flipud not needed with pcolor, instead set t & f axis: 
t = (size(YYx8,2)/Fs)*[0:size(YYx8,1)];
f = [0:M]*Fmax/(M-1);
% 4) convert to decibel scale
pxx_dB_x8      = 10*log10(pxx_reshape_x8);
subplot(4,2,8)
% Note: extend by one row & column since pcolor does not show the last row/col
P8 = [pxx_dB_x8 pxx_dB_x8(:,end); pxx_dB_x8(end,:) pxx_dB_x8(end,end)];
pcolor(t,f(3:end),P8(3:end,:)); shading flat;
% 6) choose dynamic range
% assign e.g. 60dB below peak power to the first value in the colormap
%             and the peak power to the last value in the colormap
caxis([max(max(pxx_dB_x8))-60 max(max(pxx_dB_x8))]);
% 7) select colormap
colormap(hot);
xlabel("time (minutes)",'FontSize',16);
xticks([2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70].*30)
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'})
ylabel("frequency (Hz)",'FontSize',16);
yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
grid on
axis tight


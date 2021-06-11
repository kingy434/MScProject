%%Feature matrix for tremor detection: Fundamental frequency, width of the
%%peak, height of the peak, standard deviation, 
Ts = 10;
tremor_prototype{1} = [];
tremor_prototype{2} = [];
tremor_prototype{3} = [];
tremor_prototype{4} = [];
tremor_prototype{5} = [];
tremor_prototype{6} = [];
tremor_prototype{7} = [];

tremor_prototype_features{1} = [];
tremor_prototype_features{2} = [];
tremor_prototype_features{3} = [];
tremor_prototype_features{4} = [];
tremor_prototype_features{5} = [];
tremor_prototype_features{6} = [];
tremor_prototype_features{7} = [];

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
Y_tremor_premed1 = phys(1).RW.accel(premed_tremor_start1:premed_tremor_end1, 2:4).*9.8;
Y_tremor_postmed1 = phys(1).RW.accel(postmed_tremor_start1:postmed_tremor_end1, 2:4).*9.8;
Y_tremor1 = [Y_tremor_premed1; Y_tremor_postmed1];
Y_med_ind_1 = [ones(length(Y_tremor_premed1),1); ones(length(Y_tremor_postmed1),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean1 = filter(b,a,Y_tremor1');
Y_tremor1d = downsample(TremorMean1',4);

Y_tremor_features1 = feature_matrix_all_axis(Y_tremor1d, Fs/4);

Y_med_ind_1d = downsample(Y_med_ind_1, Fs);
Y_proto1d = downsample(Y_proto1', Fs);
if length(Y_med_ind_1d)>length(Y_tremor_features1)
    Y_med_ind_1d = Y_med_ind_1d(1:end-1);
    Y_proto1d    = Y_proto1d(1:end-1);
end
Y_tremor_features1 = [ones(length(Y_tremor_features1),1), Y_tremor_features1, Y_med_ind_1d, Y_proto1d];

prototype_1 = unique(tremor_type_concat1);
for k = 1:length(prototype_1)
    ind_prototype1 = (tremor_type_concat1 == prototype_1(k));
    tremor_prototype{prototype_1(k)} = [tremor_prototype{prototype_1(k)}; tremor_data_concat1(ind_prototype1,:)];
    tremor_prototype_features{prototype_1(k)} = [tremor_prototype_features{prototype_1(k)}; feature_matrix_all_axis(tremor_data_concat1(ind_prototype1,:), Fs)];
%     tremor_prototype_features{prototype_1(k)} = feature_matrix_all_axis(tremor_data_concat1(ind_prototype1,:));
end


% Y_tremor_prototype1 = tremor_data_concat1;
% Y_tremor_prototype_features1 = feature_matrix_all_axis(Y_tremor_prototype1);
% tremor_prototype_ind1 = downsample(tremor_type_concat1,Fs);
% Y_tremor_prototype_features1 = [tremor_type_concat1, Y_tremor_prototype_features1];
%%
% Given the smooth PSD estimate, derive the same list as the short term window features
clear p s w t 
%%
Y2 = phys(6).LW.accel;
Y2l1 = labels(6).premed_tremorfinal;
Y2l2 = labels(6).postmed_tremorfinal;
Ylabel2 = [Y2l1; Y2l2];
premed_tremor_start2 = labels(6).premed_tremorstart.*Fs;
premed_tremor_end2 = labels(6).premed_tremorend.*Fs;
postmed_tremor_start2 = labels(6).postmed_tremorstart.*Fs;
postmed_tremor_end2 = labels(6).postmed_tremorend.*Fs;
Y_tremor_premed2 = phys(6).LW.accel(premed_tremor_start2:premed_tremor_end2, 2:4).*9.8;
Y_tremor_postmed2 = phys(6).LW.accel(postmed_tremor_start2:postmed_tremor_end2, 2:4).*9.8;
Y_tremor2 = [Y_tremor_premed2; Y_tremor_postmed2];
Y_med_ind_2 = [ones(length(Y_tremor_premed2),1); ones(length(Y_tremor_postmed2),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean2 = filter(b,a,Y_tremor2');
Y_tremor2d = downsample(TremorMean2',4);

Y_tremor_features2 = feature_matrix_all_axis(Y_tremor2d, Fs/4);

Y_med_ind_2d = downsample(Y_med_ind_2, Fs);
Y_proto2d = downsample(Y_proto2', Fs);
if length(Y_med_ind_2d)>length(Y_tremor_features2)
    Y_med_ind_2d = Y_med_ind_2d(1:end-1);
    Y_proto2d    = Y_proto2d(1:end-1);
end

Y_tremor_features2 = [2.*ones(length(Y_tremor_features2),1), Y_tremor_features2, Y_med_ind_2d, Y_proto2d];

prototype_2 = unique(tremor_type_concat2);
for k = 1:length(prototype_2)
    ind_prototype2 = (tremor_type_concat2 == prototype_2(k));
    tremor_prototype{prototype_2(k)} = [tremor_prototype{prototype_2(k)}; tremor_data_concat2(ind_prototype2,:)];
    tremor_prototype_features{prototype_2(k)} = [tremor_prototype_features{prototype_2(k)}; feature_matrix_all_axis(tremor_data_concat2(ind_prototype2,:), Fs)];
end
% Y_tremor_prototype2 = tremor_data_concat2;
% Y_tremor_prototype_features2 = feature_matrix_all_axis(Y_tremor_prototype2);
% tremor_prototype_ind2 = downsample(tremor_type_concat2,Fs);
% Y_tremor_prototype_features2 = [tremor_type_concat2, Y_tremor_prototype_features2];
%%
clear p s w t
Y3 = phys(8).RW.accel;
Y3l1 = labels(7).premed_tremorfinal;
Y3l2 = labels(7).postmed_tremorfinal;
Ylabel3 = [Y3l1; Y3l2];
premed_tremor_start3 = labels(7).premed_tremorstart.*Fs;
premed_tremor_end3 = labels(7).premed_tremorend.*Fs;
postmed_tremor_start3 = labels(7).postmed_tremorstart.*Fs;
postmed_tremor_end3 = labels(7).postmed_tremorend.*Fs;
Y_tremor_premed3 = phys(8).RW.accel(premed_tremor_start3:premed_tremor_end3, 2:4).*9.8;
Y_tremor_postmed3 = phys(8).RW.accel(postmed_tremor_start3:postmed_tremor_end3, 2:4).*9.8;
Y_tremor3 = [Y_tremor_premed3; Y_tremor_postmed3];
Y_med_ind_3 = [ones(length(Y_tremor_premed3),1); ones(length(Y_tremor_postmed3),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean3 = filter(b,a,Y_tremor3');
Y_tremor3d = downsample(TremorMean3',4);

Y_tremor_features3 = feature_matrix_all_axis(Y_tremor3d, Fs/4);

Y_med_ind_3d = downsample(Y_med_ind_3, Fs);
Y_proto3d = downsample(Y_proto3', Fs);
if length(Y_med_ind_3d)>length(Y_tremor_features3)
    Y_med_ind_3d = Y_med_ind_3d(1:end-1);
    Y_proto3d    = Y_proto3d(1:end-1);
end

Y_tremor_features3 = [3.*ones(length(Y_tremor_features3),1), Y_tremor_features3, Y_med_ind_3d, Y_proto3d];

prototype_3 = unique(tremor_type_concat3);
for k = 1:length(prototype_3)
    ind_prototype3 = (tremor_type_concat3 == prototype_3(k));
    tremor_prototype{prototype_3(k)} = [tremor_prototype{prototype_3(k)}; tremor_data_concat3(ind_prototype3,:)];
    tremor_prototype_features{prototype_3(k)} = [tremor_prototype_features{prototype_3(k)}; feature_matrix_all_axis(tremor_data_concat3(ind_prototype3,:))];
end
% Y_tremor_prototype3 = tremor_data_concat3;
% Y_tremor_prototype_features3 = feature_matrix_all_axis(Y_tremor_prototype3);
% tremor_prototype_ind3 = downsample(tremor_type_concat3,Fs);
% Y_tremor_prototype_features3 = [tremor_type_concat3, Y_tremor_prototype_features3];
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
Y_tremor_premed4 = phys(25).LW.accel(premed_tremor_start4:premed_tremor_end4, 2:4).*9.8;
Y_tremor_postmed4 = phys(25).LW.accel(postmed_tremor_start4:postmed_tremor_end4, 2:4).*9.8;
Y_tremor4 = [Y_tremor_premed4; Y_tremor_postmed4];
Y_med_ind_4 = [ones(length(Y_tremor_premed4),1); ones(length(Y_tremor_postmed4),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean4 = filter(b,a,Y_tremor4');
Y_tremor4d = downsample(TremorMean4',4);

Y_tremor_features4 = feature_matrix_all_axis(Y_tremor4d, Fs/4);

Y_med_ind_4d = downsample(Y_med_ind_4, Fs);
Y_proto4d = downsample(Y_proto4', Fs);
if length(Y_med_ind_4d)>length(Y_tremor_features4)
    Y_med_ind_4d = Y_med_ind_4d(1:end-1);
    Y_proto4d    = Y_proto4d(1:end-1);
end

Y_tremor_features4 = [4.*ones(length(Y_tremor_features4),1), Y_tremor_features4, Y_med_ind_4d, Y_proto4d];

% Y_tremor_prototype4 = tremor_data_concat4;
% Y_tremor_prototype_features4 = feature_matrix_all_axis(Y_tremor_prototype4);
% tremor_prototype_ind4 = downsample(tremor_type_concat4,Fs);
% Y_tremor_prototype_features4 = [tremor_type_concat4, Y_tremor_prototype_features4];

prototype_4 = unique(tremor_type_concat4);
for k = 1:length(prototype_4)
    ind_prototype4 = (tremor_type_concat4 == prototype_4(k));
    tremor_prototype{prototype_4(k)} = [tremor_prototype{prototype_4(k)}; tremor_data_concat4(ind_prototype4,:)];
    tremor_prototype_features{prototype_4(k)} = [tremor_prototype_features{prototype_4(k)}; feature_matrix_all_axis(tremor_data_concat4(ind_prototype4,:), Fs)];
end
%%
Y5 = phys(2).LW.accel;
Y5l1 = labels(19).premed_tremorfinal;
Y5l2 = labels(19).postmed_tremorfinal;
Ylabel5 = [Y5l1; Y5l2];
premed_tremor_start5 = labels(19).premed_tremorstart.*Fs;
premed_tremor_end5 = labels(19).premed_tremorend.*Fs;
postmed_tremor_start5 = labels(19).postmed_tremorstart.*Fs;
postmed_tremor_end5 = labels(19).postmed_tremorend.*Fs;
Y_tremor_premed5 = phys(2).LW.accel(premed_tremor_start5:premed_tremor_end5, 2:4).*9.8;
Y_tremor_postmed5 = phys(2).LW.accel(postmed_tremor_start5:postmed_tremor_end5, 2:4).*9.8;
Y_tremor5 = [Y_tremor_premed5; Y_tremor_postmed5];
Y_med_ind_5 = [ones(length(Y_tremor_premed5),1); ones(length(Y_tremor_postmed5),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean5 = filter(b,a,Y_tremor5');
Y_tremor5d = downsample(TremorMean5',4);

Y_tremor_features5 = feature_matrix_all_axis(Y_tremor5d, Fs/4);

Y_med_ind_5d = downsample(Y_med_ind_5, Fs);
Y_proto5d = downsample(Y_proto5', Fs);
if length(Y_med_ind_5d)>length(Y_tremor_features5)
    Y_med_ind_5d = Y_med_ind_5d(1:end-1);
    Y_proto5d    = Y_proto5d(1:end-1);
end

Y_tremor_features5 = [5.*ones(length(Y_tremor_features5),1), Y_tremor_features5, Y_med_ind_5d, Y_proto5d];

prototype_5 = unique(tremor_type_concat5);
for k = 1:length(prototype_5)
    ind_prototype5 = (tremor_type_concat5 == prototype_5(k));
    tremor_prototype{prototype_5(k)} = [tremor_prototype{prototype_5(k)}; tremor_data_concat5(ind_prototype5,:)];
    tremor_prototype_features{prototype_5(k)} = [tremor_prototype_features{prototype_5(k)}; feature_matrix_all_axis(tremor_data_concat5(ind_prototype5,:), Fs)];
end
% Y_tremor_prototype5 = tremor_data_concat5;
% Y_tremor_prototype_features5 = feature_matrix_all_axis(Y_tremor_prototype5);
% tremor_prototype_ind5 = downsample(tremor_type_concat5,Fs);
% Y_tremor_prototype_features5 = [tremor_type_concat5, Y_tremor_prototype_features5];
%%
Y6 = phys(7).RW.accel;
Y6l1 = labels(20).premed_tremorfinal;
Y6l2 = labels(20).postmed_tremorfinal;
Ylabel6 = [Y6l1; Y6l2];
premed_tremor_start6 = labels(20).premed_tremorstart.*Fs;
premed_tremor_end6 = labels(20).premed_tremorend.*Fs;
postmed_tremor_start6 = labels(20).postmed_tremorstart.*Fs;
postmed_tremor_end6 = labels(20).postmed_tremorend.*Fs;
Y_tremor_premed6 = phys(7).RW.accel(premed_tremor_start6:premed_tremor_end6, 2:4).*9.8;
Y_tremor_postmed6 = phys(7).RW.accel(postmed_tremor_start6:postmed_tremor_end6, 2:4).*9.8;
Y_tremor6 = [Y_tremor_premed6; Y_tremor_postmed6];
Y_med_ind_6 = [ones(length(Y_tremor_premed6),1); ones(length(Y_tremor_postmed6),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean6 = filter(b,a,Y_tremor6');
Y_tremor6d = downsample(TremorMean6',4);

Y_tremor_features6 = feature_matrix_all_axis(Y_tremor6d, Fs/4);


Y_med_ind_6d = downsample(Y_med_ind_6, Fs);
Y_proto6d = downsample(Y_proto6', Fs);
if length(Y_med_ind_6d)>length(Y_tremor_features6)
    Y_med_ind_6d = Y_med_ind_6d(1:end-1);
    Y_proto6d    = Y_proto6d(1:end-1);
end

Y_tremor_features6 = [6.*ones(length(Y_tremor_features6),1), Y_tremor_features6, Y_med_ind_6d, Y_proto6d];

prototype_6 = unique(tremor_type_concat6);
for k = 1:length(prototype_6)
    ind_prototype6 = (tremor_type_concat6 == prototype_6(k));
    tremor_prototype{prototype_6(k)} = [tremor_prototype{prototype_6(k)}; tremor_data_concat6(ind_prototype6,:)];
    tremor_prototype_features{prototype_6(k)} = [tremor_prototype_features{prototype_6(k)}; feature_matrix_all_axis(tremor_data_concat6(ind_prototype6,:),Fs)];
end
% Y_tremor_prototype6 = tremor_data_concat6;
% Y_tremor_prototype_features6 = feature_matrix_all_axis(Y_tremor_prototype6);
% tremor_prototype_ind6 = downsample(tremor_type_concat6,Fs);
% Y_tremor_prototype_features6 = [tremor_type_concat6, Y_tremor_prototype_features6];
%%
Y7 = phys(9).LW.accel;
Y7l1 = labels(21).premed_tremorfinal;
Y7l2 = labels(21).postmed_tremorfinal;
Ylabel7 = [Y7l1; Y7l2];
premed_tremor_start7 = labels(21).premed_tremorstart.*Fs;
premed_tremor_end7 = labels(21).premed_tremorend.*Fs;
postmed_tremor_start7 = labels(21).postmed_tremorstart.*Fs;
postmed_tremor_end7 = labels(21).postmed_tremorend.*Fs;
Y_tremor_premed7 = phys(9).LW.accel(premed_tremor_start7:premed_tremor_end7, 2:4).*9.8;
Y_tremor_postmed7 = phys(9).LW.accel(postmed_tremor_start7:postmed_tremor_end7, 2:4).*9.8;
Y_tremor7 = [Y_tremor_premed7; Y_tremor_postmed7];
Y_med_ind_7 = [ones(length(Y_tremor_premed7),1); ones(length(Y_tremor_postmed7),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
TremorMean7 = filter(b,a,Y_tremor7');
Y_tremor7d = downsample(TremorMean7',4);

Y_tremor_features7 = feature_matrix_all_axis(Y_tremor7d, Fs/4);

Y_med_ind_7d = downsample(Y_med_ind_7, Fs);
Y_proto7d = downsample(Y_proto7', Fs);
if length(Y_med_ind_7d)>length(Y_tremor_features7)
    Y_med_ind_7d = Y_med_ind_7d(1:end-1);
    Y_proto7d    = Y_proto7d(1:end-1);
end

Y_tremor_features7 = [7.*ones(length(Y_tremor_features7),1), Y_tremor_features7, Y_med_ind_7d, Y_proto7d];

prototype_7 = unique(tremor_type_concat7);
for k = 1:length(prototype_7)
    ind_prototype7 = (tremor_type_concat7 == prototype_7(k));
    tremor_prototype{prototype_7(k)} = [tremor_prototype{prototype_7(k)}; tremor_data_concat7(ind_prototype7,:)];
    tremor_prototype_features{prototype_7(k)} = [tremor_prototype_features{prototype_7(k)}; feature_matrix_all_axis(tremor_data_concat7(ind_prototype7,:),Fs)];
end
% Y_tremor_prototype7 = tremor_data_concat7;
% Y_tremor_prototype_features7 = feature_matrix_all_axis(Y_tremor_prototype7);
% tremor_prototype_ind7 = downsample(tremor_type_concat7,Fs);
% Y_tremor_prototype_features7 = [tremor_type_concat7, Y_tremor_prototype_features7];
%%
Y8 = phys(11).RW.accel;
Y8l1 = labels(22).premed_tremorfinal;
Y8l2 = labels(22).postmed_tremorfinal;
Ylabel8 = [Y8l1; Y8l2];
premed_tremor_start8 = labels(22).premed_tremorstart.*Fs;
premed_tremor_end8 = labels(22).premed_tremorend.*Fs;
postmed_tremor_start8 = labels(22).postmed_tremorstart.*Fs;
postmed_tremor_end8 = labels(22).postmed_tremorend.*Fs;
Y_tremor_premed8 = phys(11).RW.accel(premed_tremor_start8:premed_tremor_end8, 2:4).*9.8;
Y_tremor_postmed8 = phys(11).RW.accel(postmed_tremor_start8:postmed_tremor_end8, 2:4).*9.8;
Y_tremor8 = [Y_tremor_premed8; Y_tremor_postmed8];
Y_med_ind_8 = [ones(length(Y_tremor_premed8),1); ones(length(Y_tremor_postmed8),1).*2];

windowSize = 4;
b = (1/windowSize).*ones(1,windowSize);
a = 1;
WalkingFeatureMean8 = filter(b,a,Y_tremor8');
Y_tremor8d = downsample(WalkingFeatureMean8',4);

Y_tremor_features8 = feature_matrix_all_axis(Y_tremor8d, Fs/4);


Y_med_ind_8d = downsample(Y_med_ind_8, Fs);
Y_proto8d = downsample(Y_proto8', Fs);
if length(Y_med_ind_8d)>length(Y_tremor_features8)
    Y_med_ind_8d = Y_med_ind_8d(1:end-1);
    Y_proto8d    = Y_proto8d(1:end-1);
end

Y_tremor_features8 = [8.*ones(length(Y_tremor_features8),1), Y_tremor_features8, Y_med_ind_8d, Y_proto8d];

prototype_8 = unique(tremor_type_concat8);
for k = 1:length(prototype_8)
    ind_prototype8 = (tremor_type_concat8 == prototype_8(k));
    tremor_prototype{prototype_8(k)} = [tremor_prototype{prototype_8(k)}; tremor_data_concat8(ind_prototype8,:)];
    tremor_prototype_features{prototype_8(k)} = [tremor_prototype_features{prototype_8(k)}; feature_matrix_all_axis(tremor_data_concat8(ind_prototype8,:), Fs)];
end

tremor_prototype_features_matrix = [];
for k = 1:length(tremor_prototype)
    [n,d] = size(tremor_prototype_features{k});
    temp = [tremor_prototype_features{k},ones(n,1).*k]; 
    tremor_prototype_features_matrix = [tremor_prototype_features_matrix; temp];
end
% for k = 1:length(tremor_prototype)
%     tremor_prototype_features{k} = feature_matrix_all_axis(tremor_prototype{k});
% end
% Y_tremor_prototype8 = tremor_data_concat8;
% Y_tremor_prototype_features8 = feature_matrix_all_axis(Y_tremor_prototype8);
% tremor_prototype_ind8 = downsample(tremor_type_concat8,Fs);
% Y_tremor_prototype_features8 = [tremor_type_concat8, Y_tremor_prototype_features8];

% Z-scoring data coming from different individuals
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features1(:,j) = (Y_tremor_features1(:,j) - mean(Y_tremor_features1(:,j)))/std(Y_tremor_features1(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features2(:,j) = (Y_tremor_features2(:,j) - mean(Y_tremor_features2(:,j)))/std(Y_tremor_features2(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features3(:,j) = (Y_tremor_features3(:,j) - mean(Y_tremor_features3(:,j)))/std(Y_tremor_features3(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features4(:,j) = (Y_tremor_features4(:,j) - mean(Y_tremor_features4(:,j)))/std(Y_tremor_features4(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features5(:,j) = (Y_tremor_features5(:,j) - mean(Y_tremor_features5(:,j)))/std(Y_tremor_features5(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features6(:,j) = (Y_tremor_features6(:,j) - mean(Y_tremor_features6(:,j)))/std(Y_tremor_features6(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features7(:,j) = (Y_tremor_features7(:,j) - mean(Y_tremor_features7(:,j)))/std(Y_tremor_features7(:,j));
% end
% for j = 2:12 % the number of features which are real valued
%     Y_tremor_features8(:,j) = (Y_tremor_features8(:,j) - mean(Y_tremor_features8(:,j)))/std(Y_tremor_features8(:,j));
% end


Y_tremor_features = [Y_tremor_features1; Y_tremor_features2; Y_tremor_features3; Y_tremor_features4; Y_tremor_features5; Y_tremor_features6; Y_tremor_features7; Y_tremor_features8];
Ylabels = [Ylabel1; Ylabel2; Ylabel3; Ylabel4; Ylabel5; Ylabel6; Ylabel7; Ylabel8];
Ylabels = downsample(Ylabels, Fs);
Ylabels = Ylabels(3:end-3);
Ind1 = (Ylabels == '1' | Ylabels == '2');
Ylabels_num = ones(1,length(Ylabels)); 
Ylabels_num(Ind1) = Ylabels_num(Ind1) + 1;
Ylabels_num = Ylabels_num';
Ylabels_num(any(isnan(Y_tremor_features), 2), :) = [];
Y_tremor_features(any(isnan(Y_tremor_features), 2), :) = [];

for j = 2:46 % the number of features which are real valued
    Y_tremor_features(:,j) = (Y_tremor_features(:,j) - mean(Y_tremor_features(:,j)))/std(Y_tremor_features(:,j));
end
Y_tremor_features_norm = [Y_tremor_features, Ylabels_num-1];

% Ylabels = [Ylabel1; Ylabel2; Ylabel3; Ylabel4; Ylabel5; Ylabel6; Ylabel7; Ylabel8];
% Ylabels = downsample(Ylabels, Fs);
% Ylabels = Ylabels(3:end-3);
% Ind1 = (Ylabels == '1' | Ylabels == '2');
% Ylabels_num = ones(1,length(Ylabels)); 
% Ylabels_num = Ylabels_num(Ind1) + 1;


% Ind1 = (Ylabel1 == '1' | Ylabel1 == '2');
% Ind2 = (Ylabel2 == '1' | Ylabel2 == '2');
% Ind3 = (Ylabel3 == '1' | Ylabel3 == '2');
% Ind4 = (Ylabel4 == '1' | Ylabel4 == '2');
% Ind5 = (Ylabel5 == '1' | Ylabel5 == '2');
% Ind6 = (Ylabel6 == '1' | Ylabel6 == '2');
% Ind7 = (Ylabel7 == '1' | Ylabel7 == '2');
% Ind8 = (Ylabel8 == '1' | Ylabel8 == '2');
% 
figure;
subplot(4,2,1);plot(Y_tremor1(:,:));
subplot(4,2,2);plot(Y_tremor2(:,:));
subplot(4,2,3);plot(Y_tremor3(:,:));
subplot(4,2,4);plot(Y_tremor4d(:,:));
subplot(4,2,5);plot(Y_tremor5(:,:));
subplot(4,2,6);plot(Y_tremor6(:,:));
subplot(4,2,7);plot(Y_tremor7(:,:));
subplot(4,2,8);plot(Y_tremor8(:,:));

% Yx1 = Y_tremor1(:, 1);
% interval_count1 = floor(length(Yx1)/(Fs*10));% 30 seconds of data
% interval_length1 = Fs*10;
% 
% for i = 1:interval_count1
%     YYx1(i,:) = Yx1(((i-1)*interval_length1+1):i*interval_length1);
% end
% 
% for k = 1:interval_count1
%     [pxx_x1(k,:),f1] = pwelch(YYx1(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x1,2)-1)));
% %    select displayed section 
% pxx_select_x1  = pxx_x1(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x1 = transpose(pxx_select_x1);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx1,2)/Fs)*[0:size(YYx1,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x1      = 10*log10(pxx_reshape_x1);
% 
% figure;
% subplot(4,2,1)
% P1 = [pxx_dB_x1 pxx_dB_x1(:,end); pxx_dB_x1(end,:) pxx_dB_x1(end,end)];
% pcolor(t,f(3:end),P1(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x1))-60 max(max(pxx_dB_x1))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% %%
% Yx2 = Y_tremor2(Ind2, 1);
% interval_count2 = floor(length(Yx2)/(Fs*10));% 30 seconds of data
% interval_length2 = Fs*10;
% 
% for i = 1:interval_count2
%     YYx2(i,:) = Yx2(((i-1)*interval_length2+1):i*interval_length2);
% end
% 
% for k = 1:interval_count2
%     [pxx_x2(k,:),f2] = pwelch(YYx2(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x2,2)-1)));
% %    select displayed section 
% pxx_select_x2  = pxx_x2(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x2 = transpose(pxx_select_x2);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx2,2)/Fs)*[0:size(YYx2,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x2      = 10*log10(pxx_reshape_x2);
% 
% subplot(4,2,2)
% P2 = [pxx_dB_x2 pxx_dB_x2(:,end); pxx_dB_x2(end,:) pxx_dB_x2(end,end)];
% pcolor(t,f(3:end),P2(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x2))-60 max(max(pxx_dB_x2))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% 
% %%
% Yx3 = Y_tremor3(Ind3, 1);
% interval_count3 = floor(length(Yx3)/(Fs*10));% 30 seconds of data
% interval_length3 = Fs*10;
% 
% for i = 1:interval_count3
%     YYx3(i,:) = Yx3(((i-1)*interval_length3+1):i*interval_length3);
% end
% 
% for k = 1:interval_count3
%     [pxx_x3(k,:),f3] = pwelch(YYx3(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x3,2)-1)));
% %    select displayed section 
% pxx_select_x3  = pxx_x3(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x3 = transpose(pxx_select_x3);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx3,2)/Fs)*[0:size(YYx3,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x3      = 10*log10(pxx_reshape_x3);
% % 5) generate plot
% subplot(4,2,3)
% P3 = [pxx_dB_x3 pxx_dB_x3(:,end); pxx_dB_x3(end,:) pxx_dB_x3(end,end)];
% pcolor(t,f(3:end),P3(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x3))-70 max(max(pxx_dB_x3))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% 
% %%
% Yx4 = Y_tremor4(Ind4, 1);
% interval_count4 = floor(length(Yx4)/(Fs*10));% 30 seconds of data
% interval_length4 = Fs*10;
% 
% for i = 1:interval_count4
%     YYx4(i,:) = Yx4(((i-1)*interval_length4+1):i*interval_length4);
% end
% 
% for k = 1:interval_count4
%     [pxx_x4(k,:),f4] = pwelch(YYx4(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x4,2)-1)));
% %    select displayed section 
% pxx_select_x4  = pxx_x4(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x4 = transpose(pxx_select_x4);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx4,2)/Fs)*[0:size(YYx4,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x4      = 10*log10(pxx_reshape_x4);
% 
% subplot(4,2,4)
% P4 = [pxx_dB_x4 pxx_dB_x4(:,end); pxx_dB_x4(end,:) pxx_dB_x4(end,end)];
% pcolor(t,f(3:end),P4(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x4))-60 max(max(pxx_dB_x4))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% 
% %%
% Yx5 = Y_tremor5(Ind5, 1);
% interval_count5 = floor(length(Yx5)/(Fs*10));% 30 seconds of data
% interval_length5 = Fs*10;
% 
% for i = 1:interval_count5
%     YYx5(i,:) = Yx5(((i-1)*interval_length5+1):i*interval_length5);
% end
% 
% for k = 1:interval_count5
%     [pxx_x5(k,:),f5] = pwelch(YYx5(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x5,2)-1)));
% %    select displayed section 
% pxx_select_x5  = pxx_x5(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x5 = transpose(pxx_select_x5);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx5,2)/Fs)*[0:size(YYx5,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x5      = 10*log10(pxx_reshape_x5);
% 
% subplot(4,2,5)
% P5 = [pxx_dB_x5 pxx_dB_x5(:,end); pxx_dB_x5(end,:) pxx_dB_x5(end,end)];
% pcolor(t,f(3:end),P5(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x5))-60 max(max(pxx_dB_x5))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% 
% %%
% Yx6 = Y_tremor6(Ind6, 1);
% interval_count6 = floor(length(Yx6)/(Fs*10));% 30 seconds of data
% interval_length6 = Fs*10;
% 
% for i = 1:interval_count6
%     YYx6(i,:) = Yx6(((i-1)*interval_length6+1):i*interval_length6);
% end
% 
% for k = 1:interval_count6
%     [pxx_x6(k,:),f6] = pwelch(YYx6(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x6,2)-1)));
% %    select displayed section 
% pxx_select_x6  = pxx_x6(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x6 = transpose(pxx_select_x6);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx6,2)/Fs)*[0:size(YYx6,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x6      = 10*log10(pxx_reshape_x6);
% 
% subplot(4,2,6)
% P6 = [pxx_dB_x6 pxx_dB_x6(:,end); pxx_dB_x6(end,:) pxx_dB_x6(end,end)];
% pcolor(t,f(3:end),P6(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x6))-60 max(max(pxx_dB_x6))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% 
% %%
% Yx7 = Y_tremor7(Ind7, 1);
% interval_count7 = floor(length(Yx7)/(Fs*10));% 30 seconds of data
% interval_length7 = Fs*10;
% 
% for i = 1:interval_count7
%     YYx7(i,:) = Yx7(((i-1)*interval_length7+1):i*interval_length7);
% end
% 
% for k = 1:interval_count7
%     [pxx_x7(k,:),f7] = pwelch(YYx7(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x7,2)-1)));
% %    select displayed section 
% pxx_select_x7  = pxx_x7(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x7 = transpose(pxx_select_x7);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx7,2)/Fs)*[0:size(YYx7,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x7      = 10*log10(pxx_reshape_x7);
% 
% subplot(4,2,7)
% P7 = [pxx_dB_x7 pxx_dB_x7(:,end); pxx_dB_x7(end,:) pxx_dB_x7(end,end)];
% pcolor(t,f(3:end),P7(3:end,:)); shading flat;
% caxis([max(max(pxx_dB_x7))-60 max(max(pxx_dB_x7))]);
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
% 
% %%
% Yx8 = Y_tremor8(Ind8, 1);
% interval_count8 = floor(length(Yx8)/(Fs*10));% 30 seconds of data
% interval_length8 = Fs*10;
% 
% for i = 1:interval_count8
%     YYx8(i,:) = Yx8(((i-1)*interval_length8+1):i*interval_length8);
% end
% 
% for k = 1:interval_count8
%     [pxx_x8(k,:),f8] = pwelch(YYx8(k,:), 2000 , [], [], Fs);
% end
% % 1) Compute maximum frequency index
% Fmax = 15; % Hz
% M = 1 + round(Fmax/(0.5*Fs/(size(pxx_x8,2)-1)));
% %    select displayed section 
% pxx_select_x8  = pxx_x8(:,1:M);
% % 2) transpose matrix
% pxx_reshape_x8 = transpose(pxx_select_x8);
% % 3) flipud not needed with pcolor, instead set t & f axis: 
% t = (size(YYx8,2)/Fs)*[0:size(YYx8,1)];
% f = [0:M]*Fmax/(M-1);
% % 4) convert to decibel scale
% pxx_dB_x8      = 10*log10(pxx_reshape_x8);
% subplot(4,2,8)
% % Note: extend by one row & column since pcolor does not show the last row/col
% P8 = [pxx_dB_x8 pxx_dB_x8(:,end); pxx_dB_x8(end,:) pxx_dB_x8(end,end)];
% pcolor(t,f(3:end),P8(3:end,:)); shading flat;
% % 6) choose dynamic range
% % assign e.g. 60dB below peak power to the first value in the colormap
% %             and the peak power to the last value in the colormap
% caxis([max(max(pxx_dB_x8))-60 max(max(pxx_dB_x8))]);
% % 7) select colormap
% colormap(hot);
% 
% xlabel("time (minutes)",'FontSize',12);
% xticks([2 4 6 8 10 12 14 16 18 20 22 24 26].*10)
% xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12','13' })
% ylabel("frequency (Hz)",'FontSize',12);
% yticks([0.3 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);


function Y_tremor_feature = feature_matrix_all_axis(Y_tremor)
    Fs = 200;
    Ts = 10;
   
    windowSizeTime = 1.0;
    windowSize = floor(windowSizeTime*Fs);
    
    Yx = Y_tremor(:, 1);
    Num_windows = floor(length(Yx)/Fs);
    for t = 1:Num_windows
        tremor_feature_STDx(t) = std(Yx(((t-1)*Fs+1):t*Fs));
        tremor_feature_entropyx(t) = wentropy(Yx(((t-1)*Fs+1):t*Fs),'shannon');
    end
    
    Yy = Y_tremor(:, 2);
    for t = 1:Num_windows
        tremor_feature_STDy(t) = std(Yy(((t-1)*Fs+1):t*Fs));
        tremor_feature_entropyy(t) = wentropy(Yy(((t-1)*Fs+1):t*Fs),'shannon');
    end
    
    Yz = Y_tremor(:, 3);
    for t = 1:Num_windows
        tremor_feature_STDz(t) = std(Yz(((t-1)*Fs+1):t*Fs));
        tremor_feature_entropyz(t) = wentropy(Yz(((t-1)*Fs+1):t*Fs),'shannon');
    end
    
    %%Spectral features for x-axis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tremor_feature_movingSTDx = movmean(tremor_feature_STDx,Ts); %smooth feature over 10 seconds
    [sx,wx,tx] = spectrogram(Y_tremor(:,1),windowSize,0,[],Fs,'yaxis');
    for tt = 1:length(tx)
        px(:,tt) = abs(sx(:,tt));
        px(:,tt) = (px(:,tt).^2)/(windowSize.*0.5);
    end
    gait_freq_x = (wx>0.3 & wx<2);
    tremor_freq_x = (wx>4 & wx<8);
    high_tremor_freq_x = (wx>8 & wx< 12);

    [maxp_gait_x, ind_maxp_gait_x] = max(px(gait_freq_x,:));
    ind_maxp_gait_x = ind_maxp_gait_x + 1; % one smaller frequency bin which is not gait
    sump_gait_x = sum(px(gait_freq_x,:));
    [maxp_tremor_x, ind_maxp_tremor_x] = max(px(tremor_freq_x,:));
    ind_maxp_tremor_x = ind_maxp_tremor_x + 6; % six bins before the tremor range 
    sump_tremor_x = sum(px(tremor_freq_x,:));
    [maxp_hightremor_x, ind_maxp_hightremor_x] = max(px(high_tremor_freq_x,:));
    ind_maxp_hightremor_x = ind_maxp_hightremor_x + 11; % 11 bins before the high tremor range
    sump_hightremor_x = sum(px(high_tremor_freq_x,:));
    [total_maxp_x, ind_total_maxp_x] = max(px(3:end,:));
    total_sump_x = sum(px(3:end,:));
    
    se_x = pentropy(px,wx,tx); %Spectral entropy over 1 second intervals
   
    interval_count_x = floor(length(Yx)/(Fs*Ts));% 10 seconds of data
    interval_length_x = Fs*Ts;
    for i = 1:interval_count_x
        YYx(i,:) = Yx(((i-1)*interval_length_x+1):i*interval_length_x);
    end
    for k = 1:interval_count_x
        [pxx_x(k,:),f_x] = pwelch(YYx(k,:), Fs*Ts , [], [], Fs);
    end
    
    %%Spectral features for y-axis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tremor_feature_movingSTDy = movmean(tremor_feature_STDy,Ts); %smooth feature over 10 seconds
    
    [sy,wy,ty] = spectrogram(Y_tremor(:,2),windowSize,0,[],Fs,'yaxis');
    
    for tt = 1:length(ty)
        py(:,tt) = abs(sy(:,tt));
        py(:,tt) = (py(:,tt).^2)/(windowSize.*0.5);
    end
    gait_freq_y = (wy>0.3 & wy<2);
    tremor_freq_y = (wy>4 & wy<8);
    high_tremor_freq_y = (wy>8 & wy< 12);

    [maxp_gait_y, ind_maxp_gait_y] = max(py(gait_freq_y,:));
    ind_maxp_gait_y = ind_maxp_gait_y + 1; % one smaller frequency bin which is not gait
    sump_gait_y = sum(py(gait_freq_y,:));
    [maxp_tremor_y, ind_maxp_tremor_y] = max(py(tremor_freq_y,:));
    ind_maxp_tremor_y = ind_maxp_tremor_y + 6; % six bins before the tremor range 
    sump_tremor_y = sum(py(tremor_freq_y,:));
    [maxp_hightremor_y, ind_maxp_hightremor_y] = max(py(high_tremor_freq_y,:));
    ind_maxp_hightremor_y = ind_maxp_hightremor_y + 11; % 11 bins before the high tremor range
    sump_hightremor_y = sum(py(high_tremor_freq_y,:));
    [total_maxp_y, ind_total_maxp_y] = max(py(3:end,:));
    total_sump_y = sum(py(3:end,:));
    
    se_y = pentropy(py,wy,ty); %Spectral entropy over 1 second intervals
   
    interval_count_y = floor(length(Yy)/(Fs*Ts));% 10 seconds of data
    interval_length_y = Fs*Ts;
    for i = 1:interval_count_y
        YYy(i,:) = Yy(((i-1)*interval_length_y+1):i*interval_length_y);
    end
    for k = 1:interval_count_y
        [pxx_y(k,:),f_y] = pwelch(YYy(k,:), Fs*Ts , [], [], Fs);
    end
    
    %%Spectral features for z-axis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tremor_feature_movingSTDz = movmean(tremor_feature_STDz,Ts); %smooth feature over 10 seconds
    
    [sz,wz,tz] = spectrogram(Y_tremor(:,3),windowSize,0,[],Fs,'yaxis');
    
    for tt = 1:length(tz)
        pz(:,tt) = abs(sz(:,tt));
        pz(:,tt) = (pz(:,tt).^2)/(windowSize.*0.5);
    end
    gait_freq_z = (wz>0.3 & wz<2);
    tremor_freq_z = (wz>4 & wz<8);
    high_tremor_freq_z = (wz>8 & wz< 12);

    [maxp_gait_z, ind_maxp_gait_z] = max(pz(gait_freq_z,:));
    ind_maxp_gait_z = ind_maxp_gait_z + 1; % one smaller frequency bin which is not gait
    sump_gait_z = sum(pz(gait_freq_z,:));
    [maxp_tremor_z, ind_maxp_tremor_z] = max(pz(tremor_freq_z,:));
    ind_maxp_tremor_z = ind_maxp_tremor_z + 6; % six bins before the tremor range 
    sump_tremor_z = sum(pz(tremor_freq_z,:));
    [maxp_hightremor_z, ind_maxp_hightremor_z] = max(pz(high_tremor_freq_z,:));
    ind_maxp_hightremor_z = ind_maxp_hightremor_z + 11; % 11 bins before the high tremor range
    sump_hightremor_z = sum(pz(high_tremor_freq_z,:));
    [total_maxp_z, ind_total_maxp_z] = max(pz(3:end,:));
    total_sump_z = sum(pz(3:end,:));
    
    se_z = pentropy(pz,wz,tz); %Spectral entropy over 1 second intervals
   
    interval_count_z = floor(length(Yz)/(Fs*Ts));% 10 seconds of data
    interval_length_z = Fs*Ts;
    for i = 1:interval_count_z
        YYz(i,:) = Yz(((i-1)*interval_length_z+1):i*interval_length_z);
    end
    for k = 1:interval_count_z
        [pxx_z(k,:),f_z] = pwelch(YYz(k,:), Fs*Ts , [], [], Fs);
    end

    Y_tremor_feature = [tremor_feature_STDx', tremor_feature_entropyx', maxp_gait_x', sump_gait_x', maxp_tremor_x', sump_tremor_x', maxp_hightremor_x', sump_hightremor_x', total_maxp_x', total_sump_x', se_x, wx(ind_maxp_gait_x), wx(ind_maxp_tremor_x), wx(ind_maxp_hightremor_x), wx(ind_total_maxp_x), tremor_feature_STDy', tremor_feature_entropyy', maxp_gait_y', sump_gait_y', maxp_tremor_y', sump_tremor_y', maxp_hightremor_y', sump_hightremor_y', total_maxp_y', total_sump_y', se_y, wy(ind_maxp_gait_y), wy(ind_maxp_tremor_y), wy(ind_maxp_hightremor_y), wy(ind_total_maxp_y), tremor_feature_STDz', tremor_feature_entropyz', maxp_gait_z', sump_gait_z', maxp_tremor_z', sump_tremor_z', maxp_hightremor_z', sump_hightremor_z', total_maxp_z', total_sump_z', se_z, wz(ind_maxp_gait_z), wz(ind_maxp_tremor_z), wz(ind_maxp_hightremor_z), wz(ind_total_maxp_z)];
end

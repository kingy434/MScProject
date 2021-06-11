from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat_data = 'Tremor_matrix_with_causal_factors_and_prototypes.mat'
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['Y_tremor_features_norm'];
    
#Y_features_whiten = Y_features[:,1:46]
#Y_features_whiten = Y_features_whiten - np.mean(Y_features_whiten,axis=0)
#from sklearn.decomposition import PCA
#pca = PCA(n_components=45)
#pca.fit(Y_features_whiten)
#Y_features_whiten = pca.fit_transform(Y_features_whiten)
#Y_features[:,1:46] = Y_features_whiten

data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])

data_features = data_features.dropna()
data_features = data_features.drop(columns=["Prototype_ID", "Activity_label", "Medication_Intake"])

IDlist = {'1', '2', '3', '4', '5', '6', '7', '8'}

outcomevar = 'Non-tremor/Tremor'
outcomevar_id = 48
idcolumn = 'ID'
idcolumn_id = 1

# Initialize empty lists and dataframe 
errors = []
predictions = []
prob_predictions = []
errors = []
labels = []

# Run LOOCV Random Forest! 
for i in range(8):
    pred, prob_pred, error, label = RFLOOCV(data_features, i+1, outcomevar, idcolumn)
    predictions.append(pred)
    prob_predictions.append(prob_pred)
    errors.append(error)
    labels.append(label)
    idt = str(i)
    print('...' + idt + ' processing complete.')
    

import pickle
with open('RF_tremor_classifier_leave_one_subject.pickle', 'wb') as f:
    pickle.dump([prob_predictions, labels], f)
    
from sklearn import metrics
import seaborn as sns

#fmri = sns.load_dataset("fmri")
#sns.relplot(
#    data=fmri, kind="line",
#    x="timepoint", y="signal", col="region",
#    hue="event", style="event",
#)

#fpr_tpr_plot = pd.DataFrame(data=np.vstack((fpr,tpr)).T, columns["fpr", "tpr"]) 
fig, ax = plt.subplots()
#sns.set_theme()

for j in range(8):
    ind_true = np.where(labels[j] == 1)
    ind_false = np.where(labels[j] == 0) 
    fpr, tpr, thresholds = metrics.roc_curve(labels[j], prob_predictions[j][:,1])
    data_roc = np.vstack((fpr,tpr)).T
    fpr_plot = pd.DataFrame(data=data_roc, columns=["fpr", "tpr"])
    # This is the ROC curve
   # fpr_tpr_plot = pd.DataFrame(data=data_roc, columns["fpr", "tpr"]) 
    sns.set()
    sns.lineplot(data=fpr_plot,  x="fpr", y="tpr", ax=ax)
    
   # plt.plot(fpr,tpr)
#sns.plt.show()

balanced_accuracy = (tpr+(1-fpr))/2

max_val = max(balanced_accuracy.reshape(-1,1))
ind_max = np.argmax(balanced_accuracy.reshape(-1,1))
sensitivity = tpr[ind_max]
specificity = 1 - fpr[ind_max]
thr = thresholds[ind_max]

with open('RF_tremor_classifier_leave_one_subject.pickle', 'wb') as f:
    pickle.dump([prob_predictions, labels, thr], f)
    

fig, ax = plt.subplots()
####10-fold cross validation without adjusting for variation in individual identity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor"])

X = data_features.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor"]).to_numpy()
y = data_features["Non-tremor/Tremor"].to_numpy()

prob_predictions_10fold = []
index_test = []
balanced_accuracy_10fold = []
sensitivity_10fold = []
specificity_10fold = []
thr = []

numestimators=100
indecies = np.arange(len(X))
for train_state in range(100):
    X_train, X_test, y_train, y_test, indecies_train, indecies_test = train_test_split(X, y, indecies, test_size=0.1, random_state=train_state)

    rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
    # Train the model on training data
    rf.fit(X_train, y_train);
    predictions_10fold = rf.predict(X_test)
    prob_predictions_10fold_temp = rf.predict_proba(X_test)
    prob_predictions_10fold.append(prob_predictions_10fold_temp)
    index_test.append(indecies_test)
    ind_true = np.where(y_test == 1)
    ind_false = np.where(y_test == 0) 
    fpr_10fold, tpr_10fold, thresholds = metrics.roc_curve(y_test, prob_predictions_10fold_temp[:,1])

    balanced_accuracy_10fold_temp = (tpr_10fold+(1-fpr_10fold))/2
    balanced_accuracy_10fold.append((tpr_10fold+(1-fpr_10fold))/2)

    max_val = max(balanced_accuracy_10fold_temp.reshape(-1,1))
    ind_max = np.argmax(balanced_accuracy_10fold_temp.reshape(-1,1))
    sensitivity_10fold.append(tpr_10fold[ind_max])
    specificity_10fold.append(1 - fpr_10fold[ind_max])
    thr.append(thresholds[ind_max])

    
    data_roc = np.vstack((fpr_10fold,tpr_10fold)).T
    fpr_plot_10fold = pd.DataFrame(data=data_roc, columns=["fpr", "tpr"])
    sns.set()
    sns.lineplot(data=fpr_plot_10fold,  x="fpr", y="tpr", ax=ax)
    #plt.plot(fpr_10fold, tpr_10fold)
    
predicted_prob = np.zeros((len(ind_ID1[0]),2))
for i in range(100):
    for j in range(len(ind_ID1[0])):
        if ind_ID1[0][j] in index_test[i]:
            loc_index = np.where(ind_ID1[0][j] == index_test[i])
            predicted_prob[j,:] = prob_predictions_10fold[i][int(np.array(loc_index))]
            

predicted_prob_per_ID = []
thresholds_per_ID = []
for ii in range(8):
    ind_ID = np.array(np.where(data_features["ID"] == ii+1))
    predicted_prob = np.zeros((len(ind_ID[0]),2))
    for i in range(100):
        for j in range(len(ind_ID[0])):
            if ind_ID[0][j] in index_test[i]:
                loc_index = np.where(ind_ID[0][j] == index_test[i])
                predicted_prob[j,:] = prob_predictions_10fold[i][int(np.array(loc_index))]
    fpr_10fold_ID, tpr_10fold_ID, thresholds_ID = metrics.roc_curve(labels[ii] , predicted_prob[:,1])
    balanced_accuracy_10fold_temp = (tpr_10fold_ID+(1-fpr_10fold_ID))/2
    max_val = max(balanced_accuracy_10fold_temp.reshape(-1,1))
    ind_max = np.argmax(balanced_accuracy_10fold_temp.reshape(-1,1))
    thresholds_per_ID.append(thresholds_ID[ind_max])
    predicted_prob_per_ID.append(predicted_prob)
    
with open('RF_tremor_classifier_10fold_crossvalidation_per_ID.pickle', 'wb') as f:
    pickle.dump([predicted_prob_per_ID, thresholds_per_ID, thr], f)

balanced_accuracy_10fold_perID = []
sensitivity_10fold_perID = []
specificity_10fold_perID = []
thr_perID = []

fpr_10fold_ID1, tpr_10fold_ID1, thresholds_ID1 = metrics.roc_curve(labels[0] , predicted_prob[:,1])
fpr_10fold_ID2, tpr_10fold_ID2, thresholds_ID2 = metrics.roc_curve(labels[1] , predicted_prob[:,1])
fpr_10fold_ID3, tpr_10fold_ID3, thresholds_ID3 = metrics.roc_curve(labels[2] , predicted_prob[:,1])
fpr_10fold_ID4, tpr_10fold_ID4, thresholds_ID4 = metrics.roc_curve(labels[3] , predicted_prob[:,1])
fpr_10fold_ID5, tpr_10fold_ID5, thresholds_ID5 = metrics.roc_curve(labels[4] , predicted_prob[:,1])
fpr_10fold_ID6, tpr_10fold_ID6, thresholds_ID6 = metrics.roc_curve(labels[5] , predicted_prob[:,1])
fpr_10fold_ID7, tpr_10fold_ID7, thresholds_ID7 = metrics.roc_curve(labels[6] , predicted_prob[:,1])
fpr_10fold_ID8, tpr_10fold_ID8, thresholds_ID8 = metrics.roc_curve(labels[7] , predicted_prob[:,1])
balanced_accuracy_10fold_temp = (tpr_10fold_ID8+(1-fpr_10fold_ID8))/2
balanced_accuracy_10fold_perID.append((tpr_10fold_ID8+(1-fpr_10fold_ID8))/2)

max_val = max(balanced_accuracy_10fold_temp.reshape(-1,1))
ind_max = np.argmax(balanced_accuracy_10fold_temp.reshape(-1,1))
sensitivity_10fold_perID.append(tpr_10fold_ID8[ind_max])
specificity_10fold_perID.append(1 - fpr_10fold_ID8[ind_max])
thr_perID.append(thresholds_ID8[ind_max])
            
    
import pickle
with open('RF_tremor_classifier_10fold_crossvalidation.pickle', 'wb') as f:
    pickle.dump([prob_predictions_10fold, index_test, balanced_accuracy_10fold, sensitivity_10fold, specificity_10fold, thr], f)
    
data_post_classified = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor"])



from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mat_data = 'Tremor_matrix_with_causal_factors_and_prototypes.mat'
mat_contents = sio.loadmat(mat_data)
Y_features = mat_contents['Y_tremor_features_norm'];
    
data_features = pd.DataFrame(data=Y_features, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])

###Adjustment for factors####
# Estimating the empirical probabilities of the different factors
# Age: 58 69 59 69 63 59 49 72 ---> categorical of above and below 65
Age = np.ones(len(data_features))
Age[data_features["ID"] == 1] = 1
Age[data_features["ID"] == 2] = 2
Age[data_features["ID"] == 3] = 1
Age[data_features["ID"] == 4] = 2
Age[data_features["ID"] == 5] = 1
Age[data_features["ID"] == 6] = 1
Age[data_features["ID"] == 7] = 1
Age[data_features["ID"] == 8] = 2
# Time since diagnosis: 8 10 6 8 6 4 5 13 --->  Categorical of more or less then 7 years since diagnosis 
TimeDiagnose = np.ones(len(data_features))
TimeDiagnose[data_features["ID"] == 1] = 2
TimeDiagnose[data_features["ID"] == 2] = 2
TimeDiagnose[data_features["ID"] == 3] = 1
TimeDiagnose[data_features["ID"] == 4] = 2
TimeDiagnose[data_features["ID"] == 5] = 1
TimeDiagnose[data_features["ID"] == 6] = 1
TimeDiagnose[data_features["ID"] == 7] = 1
TimeDiagnose[data_features["ID"] == 8] = 2
# Probability of tremor ON per age group p(T|A)

# Categorize tremor prototypes combine resting on surface (option 1)
proto_len = np.sum(data_features["Prototype_ID"] != 0)
Tremor_proto = np.zeros(len(data_features))
Tremor_proto[data_features["Prototype_ID"] == 1] = 1
Tremor_proto[data_features["Prototype_ID"] == 2] = 1
Tremor_proto[data_features["Prototype_ID"] == 3] = 2
Tremor_proto[data_features["Prototype_ID"] == 4] = 2
Tremor_proto[data_features["Prototype_ID"] == 5] = 3
Tremor_proto[data_features["Prototype_ID"] == 6] = 4
Tremor_proto[data_features["Prototype_ID"] == 7] = 4

Tremor_arm_rest = np.zeros(len(data_features))
Tremor_arm_rest[data_features["Prototype_ID"] == 1] = 1
Tremor_arm_rest[data_features["Prototype_ID"] == 3] = 1
Tremor_arm_rest[data_features["Prototype_ID"] == 6] = 1

Tremor_arm_rest[data_features["Prototype_ID"] == 2] = 2
Tremor_arm_rest[data_features["Prototype_ID"] == 4] = 2
Tremor_arm_rest[data_features["Prototype_ID"] == 7] = 2

Tremor_arm_rest[data_features["Prototype_ID"] == 5] = 3
#plt.show()

###Find unknown prototypes via classification
Ind_proto_gait_known = data_features["Activity_label"] == 3
data_features["Prototype_ID"][Ind_proto_gait_known] = 5

Ind_proto_known = data_features["Prototype_ID"] != 0
X_proto_train = data_features[Ind_proto_known]
X_proto_train = X_proto_train.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor", "Activity_label"])
y_proto_train = data_features["Prototype_ID"][Ind_proto_known]

Ind_proto_unknown = data_features["Prototype_ID"] == 0
X_proto_test = data_features[Ind_proto_unknown]
X_proto_test = X_proto_test.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor", "Activity_label"])

rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
# Train the model on training data
rf.fit(X_proto_train, y_proto_train);
predictions_proto = rf.predict(X_proto_test)
prob_predictions_proto = rf.predict_proba(X_proto_test)
y_proto = data_features["Prototype_ID"]
y_proto[Ind_proto_unknown] = predictions_proto
data_features["Prototype_ID"][Ind_proto_unknown] = predictions_proto

y_proto_group_train = Tremor_proto
y_proto_group_train = y_proto_group_train[Ind_proto_known]
rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
rf.fit(X_proto_train, y_proto_group_train);
predictions_proto_group = rf.predict(X_proto_test)
prob_predictions_proto_group = rf.predict_proba(X_proto_test)
y_proto_group = Tremor_proto
y_proto_group[Ind_proto_unknown] = predictions_proto_group
Tremor_proto[Ind_proto_unknown] = predictions_proto_group


y_arm_rest_train = Tremor_arm_rest
y_arm_rest_train = y_arm_rest_train[Ind_proto_known]
rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
rf.fit(X_proto_train, y_arm_rest_train);
predictions_arm_rest = rf.predict(X_proto_test)
prob_predictions_arm_rest = rf.predict_proba(X_proto_test)
y_arm_rest = Tremor_arm_rest
y_arm_rest[Ind_proto_unknown] = predictions_arm_rest
Tremor_arm_rest[Ind_proto_unknown] = predictions_arm_rest
    


#f_proto1 = np.sum(data_features["Prototype_ID"] == 1 or data_features["Prototype_ID"] == 2)
#f_proto2 = np.sum(data_features["Prototype_ID"] == 3 or data_features["Prototype_ID"] == 4)
#f_proto3 = np.sum(data_features["Prototype_ID"] == 5)
#f_proto4 = np.sum(data_features["Prototype_ID"] == 6 or data_features["Prototype_ID"] == 7)

#f_arm_resting1 = np.sum(data_features["Prototype_ID"] == 1 or data_features["Prototype_ID"] == 3 or data_features["Prototype_ID"] == 6)
#f_arm_resting2 = np.sum(data_features["Prototype_ID"] == 2 or data_features["Prototype_ID"] == 4 or data_features["Prototype_ID"] == 7)
Tremor_on_off = np.zeros(len(data_features))
Tremor_on_off[data_features["Non-tremor/Tremor"]==1] = 1
Tremor_on_off[data_features["Non-tremor/Tremor"]==0] = 0


p_med_n = np.zeros(len(data_features))
p_age_n = np.zeros(len(data_features))
p_proto_given_t_n = np.zeros(len(data_features))
p_p_m_t = np.zeros(len(data_features))
p_p_a_m_t = np.zeros(len(data_features))
p_t_given_a_n = np.zeros(len(data_features))
w_n = np.zeros(len(data_features))

for n in range(len(data_features)):
    p_med_n[n] = np.sum(data_features["Medication_Intake"] == data_features["Medication_Intake"][n])/len(data_features)    
    p_age_n[n] = np.sum(Age == Age[n])/len(data_features)
    p_t_given_a_n[n] = np.sum(Tremor_on_off[Age == Age[n]] == Tremor_on_off[n])/np.sum(Tremor_on_off == Tremor_on_off[n])
    p_proto_given_t_n[n] = np.sum(Tremor_proto[Tremor_on_off == Tremor_on_off[n]] == Tremor_proto[n])/np.sum(Tremor_on_off == Tremor_on_off[n])
    ind_med = data_features["Medication_Intake"][Tremor_proto==Tremor_proto[n]] == data_features["Medication_Intake"][n]
    data_temp_proto = data_features["Non-tremor/Tremor"][Tremor_proto==Tremor_proto[n]]
    p_p_m_t[n] = np.sum(data_temp_proto[ind_med] == Tremor_on_off[n])/np.sum(data_features["Non-tremor/Tremor"] == Tremor_on_off[n])
    age_temp = Age[Tremor_proto==Tremor_proto[n]]
    age_temp2 = age_temp[ind_med]
    data_temp_proto = data_temp_proto[ind_med]
    if np.sum(data_temp_proto[age_temp2 == Age[n]]==1) == 0:
        print('Hey')
        #print(np.sum(age_temp[ind_med] == Age[n]))
        print(Tremor_proto[n])
        print(Age[n])
        print(data_features["Medication_Intake"][n])
        print(data_features["Non-tremor/Tremor"][n])
    p_p_a_m_t[n] = np.sum(data_temp_proto[age_temp2 == Age[n]]==Tremor_on_off[n])/np.sum(data_features["Non-tremor/Tremor"] == Tremor_on_off[n])
    w_n[n] = p_med_n[n]*p_age_n[n]*p_t_given_a_n[n]*p_proto_given_t_n[n]/p_p_m_t[n]# no a *p_age_n[n]
    
    
Tremor_arm_rest
Tremor_activity = data_features["Activity_label"]
Tremor_activity[np.isnan(Tremor_activity)] = 0
TimeDiagnose
Activity (sitting standing)

p_med_n = np.zeros(len(data_features))
p_age_n = np.zeros(len(data_features))
p_proto_given_t_n = np.zeros(len(data_features))
p_p_m_t = np.zeros(len(data_features))
p_p_a_m_t = np.zeros(len(data_features))
p_t_given_a_n = np.zeros(len(data_features))
w_n = np.zeros(len(data_features))
p_arm_free_n = np.zeros(len(data_features))
p_activity = np.zeros(len(data_features))
p_t_given_a_p_arm_free_p_activity_n = np.zeros(len(data_features))
p_Pr_Ps_a_t = np.zeros(len(data_features))

#####Alternative causal structure    
for n in range(len(data_features)):
    #p_med_n[n] = np.sum(data_features["Medication_Intake"] == data_features["Medication_Intake"][n])/len(data_features)    
    p_age_n[n] = np.sum(Age == Age[n])/len(data_features)
    p_arm_free_n[n] = np.sum(Tremor_arm_rest == Tremor_arm_rest[n])/len(data_features)
    p_activity[n] = np.sum(Tremor_activity == Tremor_activity[n])/len(data_features)
    p_t_given_a_p_arm_free_p_activity_n[n] = np.sum(Tremor_on_off[Age == Age[n]] == Tremor_on_off[n])/np.sum(Tremor_on_off == Tremor_on_off[n])
       
  #  p_proto_given_t_n[n] = np.sum(Tremor_proto[Tremor_on_off == Tremor_on_off[n]] == Tremor_proto[n])/np.sum(Tremor_on_off == Tremor_on_off[n])
    ind_activity = Tremor_activity[Tremor_arm_rest==Tremor_arm_rest[n]] == Tremor_activity[n]
    Tremor_activity_temp = Tremor_activity[Tremor_arm_rest==Tremor_arm_rest[n]]  
    
    #ind_med_med = Tremor_activity_temp[ind_activity]==Tremor_activity[n]
      
    data_temp_proto = data_features["Non-tremor/Tremor"][Tremor_arm_rest==Tremor_arm_rest[n]]
    data_temp_proto = data_temp_proto[ind_activity]
   # data_temp_proto = data_temp_proto[ind_activity]
   # p_p_m_t[n] = np.sum(data_temp_proto[ind_med] == Tremor_on_off[n])/np.sum(data_features["Non-tremor/Tremor"] == Tremor_on_off[n])
    age_temp = Age[Tremor_arm_rest==Tremor_arm_rest[n]]
    age_temp = age_temp[ind_activity]   
   # age_temp2 = age_temp[ind_activity]    
    
    p_Pr_Ps_a_t[n] = np.sum(data_temp_proto[age_temp == Age[n]]==Tremor_on_off[n])/np.sum(data_features["Non-tremor/Tremor"] == Tremor_on_off[n])
    #data_temp_proto = data_temp_proto[ind_med_med]
    
    
    
    if np.sum(data_temp_proto[age_temp == Age[n]] == Tremor_on_off[n]) == 0:
        print('Hey')
        #print(np.sum(age_temp[ind_med] == Age[n]))
        print(Tremor_proto[n])
        print(Age[n])
        print(data_features["Medication_Intake"][n])
        print(data_features["Non-tremor/Tremor"][n])
    #p_a_m_t[n] = np.sum(data_temp_proto[age_temp2 == Age[n]]==Tremor_on_off[n])/np.sum(data_features["Non-tremor/Tremor"] == Tremor_on_off[n])
    w_n[n] = p_age_n[n]*p_t_given_a_p_arm_free_p_activity_n[n]*p_activity[n]*p_arm_free_n[n]/p_Pr_Ps_a_t[n]
    # p_proto_given_t_n = np.sum(data_features["Prototype_ID"]==data_features["Prototype_ID"][n] + data_features["Non-tremor/Tremor"]==1)
    # p_proto_group_given_t_n = np.sum(Tremor_proto==Tremor_proto[n] and data_features["Non-tremor/Tremor"]==1)
    # p_proto_arm_given_t_n = np.sum(Tremor_arm_rest==Tremor_arm_rest[n] and data_features["Non-tremor/Tremor"]==1)
    # p_p_a_m_t = 
    # p_p_a_m_t = np.sum(Tremor_proto==Tremor_proto[n] and Age == Age[n] and data_features["Medication_Intake"] == data_features["Medication_Intake"][n] and data_features["Non-tremor/Tremor"])
#probabilities[:, :, :] = w_n_weighted)

#n1, n2, m = probabilities.shape

#cum_prob = np.cumsum(probabilities, axis=-1) # shape (n1, n2, m)
#r = np.random.uniform(size=(n1, n2, 1))

# argmax finds the index of the first True value in the last axis.
#samples = np.argmax(cum_prob > r, axis=-1)

n_sample = len(data_features)
ind_data_bootstrap = np.random.choice(range(len(data_features)), size = n_sample, replace=True, p = w_n/np.sum(w_n))
     
Y_features_controlled = Y_features[ind_data_bootstrap,:]
data_features = pd.DataFrame(data=Y_features_controlled, columns=["ID", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8", "Col9", "Col10","Col11", "Col12", "Col13", "Col14", "Col15", "Col16", "Col17", "Col18", "Col19", "Col20","Col21", "Col22", "Col23", "Col24", "Col25", "Col26", "Col27", "Col28", "Col29", "Col30","Col31", "Col32", "Col33", "Col34", "Col35", "Col36", "Col37", "Col38", "Col39", "Col40","Col41", "Col42", "Col43", "Col44", "Col45", "Col46","Medication_Intake","Prototype_ID","Non-tremor/Tremor","Activity_label"])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



data_features = data_features.dropna()
data_features = data_features.drop(columns=["Prototype_ID","Medication_Intake", "Activity_label"])


IDlist = {'1', '2', '3', '4', '5', '6', '7', '8'}

outcomevar = 'Non-tremor/Tremor'
idcolumn = 'ID'
idcolumn_id = 1
outcomevar_id = 47

# Initialize empty lists and dataframe 
errors = []
predictions = []
prob_predictions = []
errors = []
labels = []

# Run LOOCV Random Forest! 
for i in range(8):
    pred, prob_pred, error, label = RFLOOCV(data_features, i+1, outcomevar, idcolumn)
    predictions.append(pred)
    prob_predictions.append(prob_pred)
    errors.append(error)
    labels.append(label)
    idt = str(i)
    print('...' + idt + ' processing complete.')
    

#import pickle
#with open('RF_tremor_classifier_leave_one_subject.pickle', 'wb') as f:
#    pickle.dump([prob_predictions, labels], f)
    
from sklearn import metrics
import seaborn as sns

fig, ax = plt.subplots()
sensitivity = np.zeros(8,)
specificity = np.zeros(8,)
for j in range(8):
    ind_true = np.where(labels[j] == 1)
    ind_false = np.where(labels[j] == 0) 
    fpr, tpr, thresholds = metrics.roc_curve(labels[j], prob_predictions[j][:,1])
    data_roc = np.vstack((fpr,tpr)).T
    fpr_plot = pd.DataFrame(data=data_roc, columns=["fpr", "tpr"])
    # This is the ROC curve
    sns.set()
    sns.lineplot(data=fpr_plot,  x="fpr", y="tpr", ax=ax)
    balanced_accuracy = (tpr+(1-fpr))/2
    max_val = max(balanced_accuracy.reshape(-1,1))
    ind_max = np.argmax(balanced_accuracy.reshape(-1,1))
    sensitivity[j] = tpr[ind_max]
    specificity[j] = 1 - fpr[ind_max]
  #  print(sensitivity)
    #print(specificity)
    
    
balanced_accuracy = (tpr+(1-fpr))/2

max_val = max(balanced_accuracy.reshape(-1,1))
ind_max = np.argmax(balanced_accuracy.reshape(-1,1))
sensitivity = tpr[ind_max]
specificity = 1 - fpr[ind_max]
thr = thresholds[ind_max]    
    
    
    
    
    
X = data_features.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor", "Activity_label"]).to_numpy()
y = data_features["Non-tremor/Tremor"].to_numpy()

numestimators=100
for train_state in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=train_state)

    rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
    # Train the model on training data
    rf.fit(X_train, y_train);
    predictions_10fold = rf.predict(X_test)
    prob_predictions_10fold = rf.predict_proba(X_test)

    ind_true = np.where(y_test == 1)
    ind_false = np.where(y_test == 0) 
    fpr_10fold, tpr_10fold, thresholds = metrics.roc_curve(y_test, prob_predictions_10fold[:,1])

    balanced_accuracy_10fold = (tpr_10fold+(1-fpr_10fold))/2

    max_val = max(balanced_accuracy_10fold.reshape(-1,1))
    ind_max = np.argmax(balanced_accuracy_10fold.reshape(-1,1))
    sensitivity_10fold_controlled = tpr_10fold[ind_max]
    specificity_10fold_controlled = 1 - fpr_10fold[ind_max]
    
    data_roc_controlled = np.vstack((fpr_10fold,tpr_10fold)).T
    fpr_plot_10fold_controlled = pd.DataFrame(data=data_roc_controlled, columns=["fpr", "tpr"])
    sns.set()
    sns.lineplot(data=fpr_plot_10fold_controlled,  x="fpr", y="tpr", ax=ax)
    # p_p_a_m_t = np.sum(Tremor_proto==Tremor_proto[n] and Age == Age[n] and data_features["Medication_Intake"] == data_features["Medication_Intake"][n] and data_features["Non-tremor/Tremor"])


prob_predictions_10fold = []
index_test = []
balanced_accuracy_10fold = []
sensitivity_10fold = []
specificity_10fold = []
thr = []

X = data_features.drop(columns=["Prototype_ID", "Medication_Intake", "ID", "Non-tremor/Tremor", "Activity_label"]).to_numpy()
y = data_features["Non-tremor/Tremor"].to_numpy()

numestimators=100
indecies = np.arange(len(X))
for train_state in range(100):
    X_train, X_test, y_train, y_test, indecies_train, indecies_test = train_test_split(X, y, indecies, test_size=0.1, random_state=train_state)

    rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
    # Train the model on training data
    rf.fit(X_train, y_train);
    predictions_10fold = rf.predict(X_test)
    prob_predictions_10fold_temp = rf.predict_proba(X_test)
    prob_predictions_10fold.append(prob_predictions_10fold_temp)
    index_test.append(indecies_test)
    ind_true = np.where(y_test == 1)
    ind_false = np.where(y_test == 0) 
    fpr_10fold, tpr_10fold, thresholds = metrics.roc_curve(y_test, prob_predictions_10fold_temp[:,1])

    balanced_accuracy_10fold_temp = (tpr_10fold+(1-fpr_10fold))/2
    balanced_accuracy_10fold.append((tpr_10fold+(1-fpr_10fold))/2)

    max_val = max(balanced_accuracy_10fold_temp.reshape(-1,1))
    ind_max = np.argmax(balanced_accuracy_10fold_temp.reshape(-1,1))
    sensitivity_10fold.append(tpr_10fold[ind_max])
    specificity_10fold.append(1 - fpr_10fold[ind_max])
    thr.append(thresholds[ind_max])

    
    data_roc = np.vstack((fpr_10fold,tpr_10fold)).T
    fpr_plot_10fold = pd.DataFrame(data=data_roc, columns=["fpr", "tpr"])
    sns.set()
    sns.lineplot(data=fpr_plot_10fold,  x="fpr", y="tpr", ax=ax)
    
# Probability of prototype given tremor or given tremor and gait
# What is the probability of prototype groups 1 2 3 given tremor
# What is the probability of arm free/resting on surface given a tremor    
#P(prototype, age, medication intake, T=tremor)    



def RFLOOCV(data, ids, outcomevar, idcolumn, numestimators=100, fs=0.02):
    """
        Intermediate function. 
            
    """
    # Get important features 
    #listimportances = LOOCV_featureselection(data, ids, outcomevar, dropcols, idcolumn, numestimators)
    #filteredi = listimportances[listimportances['importances'] < fs]
    #filteredi = filteredi['value']
    LOOCV_O = ids
    #data[idcolumn] = data[idcolumn].apply(str)
    data_filtered = data[data[idcolumn] != LOOCV_O]
    data_cv = data[data[idcolumn] == LOOCV_O]
   
    # Test data - the person left out of training
    data_test = data_cv.drop(columns=idcolumn)
   # data_test = data_test.drop(columns=filteredi) #cvf
    X_test = data_test.drop(columns=[outcomevar])
    y_test = data_test[outcomevar] #This is the outcome variable
    
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=idcolumn)
    #data_train = data_train.drop(columns=filteredi)
    X_train = data_train.drop(columns=[outcomevar])
    
    feature_list = list(X_train.columns)
    X_train= np.array(X_train)
    y_train = np.array(data_train[outcomevar]) #Outcome variable here

    
    from sklearn.ensemble import RandomForestClassifier
    # Instantiate model with numestimators decision trees
    rf = RandomForestClassifier(n_estimators = numestimators, random_state = 0, class_weight="balanced")
    # Train the model on training data
    rf.fit(X_train, y_train);
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)
    prob_predictions = rf.predict_proba(X_test)
    errors = abs(predictions - y_test)
    labels = y_test
    
    return predictions, prob_predictions, errors, labels
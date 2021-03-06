#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA

# %%
# os.chdir("C:\won\data\seperate_garbage")
os.chdir("/Users/wonhyung64/data/seperate")
"/Users/wonhyung64/data/House hold solid waste segregation  data.xls"
data = pd.read_csv("seperate.csv", encoding='utf-8')
# %% Work 1
df_q7 = data["Mixedwaste"]
q7_yes = len(df_q7[(df_q7 == 1)==1])
q7_no = len(df_q7[(df_q7 == 1)==0])
print("[Q7] yes : {}, no : {}".format(round(q7_yes / (q7_yes + q7_no),3), round(q7_no / (q7_yes + q7_no), 3)))

df_q6 = data["Wastemanagement"]
q6_yes = len(df_q6[(df_q6 == 1)==1])
q6_no = len(df_q6[(df_q6 == 1)==0])
print("[Q6] yes : {}, no : {}".format(round(q6_yes / (q6_yes + q6_no),3), round(q6_no / (q6_yes + q6_no), 3)))

df_q11 = data["Segregation"]
q11_yes = len(df_q11[(df_q11 == 1)==1])
q11_no = len(df_q11[(df_q11 == 1)==0])
print("[Q11] yes : {}, no : {}".format(round(q11_yes / (q11_yes + q11_no),3), round(q11_no / (q11_yes + q11_no), 3)))

#%%
index = ['Q7','Q6','Q11']
columns = ['Yes','No']

data_ = np.array([[round(q7_yes / (q7_yes + q7_no),3), round(q7_no / (q7_yes + q7_no), 3)],
                [round(q6_yes / (q6_yes + q6_no),3), round(q6_no / (q6_yes + q6_no), 3)],
                [round(q11_yes / (q11_yes + q11_no),3), round(q11_no / (q11_yes + q11_no), 3)]
                ])

work1 = pd.DataFrame(data_, index=index, columns=columns)
print(work1)
#%% Work 2
df_q14_19 = data.iloc[:,13:19]

q14_5 = len(df_q14_19[(df_q14_19["Absence"] == 5)])
q14_4 = len(df_q14_19[(df_q14_19["Absence"] == 4)])
q14_3 = len(df_q14_19[(df_q14_19["Absence"] == 3)])
q14_2 = len(df_q14_19[(df_q14_19["Absence"] == 2)])
q14_1 = len(df_q14_19[(df_q14_19["Absence"] == 1)])
print("[Q14] A5 : {}, A4 : {}, A3 : {}, A2 : {}, A1 : {}".format(
    round(q14_5 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3), 
    round(q14_4 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3),
    round(q14_3 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3),
    round(q14_2 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3),
    round(q14_1 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3))
)

q15_5 = len(df_q14_19[(df_q14_19["Awareness"] == 5)])
q15_4 = len(df_q14_19[(df_q14_19["Awareness"] == 4)])
q15_3 = len(df_q14_19[(df_q14_19["Awareness"] == 3)])
q15_2 = len(df_q14_19[(df_q14_19["Awareness"] == 2)])
q15_1 = len(df_q14_19[(df_q14_19["Awareness"] == 1)])
print("[Q15] A5 : {}, A4 : {}, A3 : {}, A2 : {}, A1 : {}".format(
    round(q15_5 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3), 
    round(q15_4 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3),
    round(q15_3 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3),
    round(q15_2 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3),
    round(q15_1 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3))
)

q16_5 = len(df_q14_19[(df_q14_19["Shortage"] == 5)])
q16_4 = len(df_q14_19[(df_q14_19["Shortage"] == 4)])
q16_3 = len(df_q14_19[(df_q14_19["Shortage"] == 3)])
q16_2 = len(df_q14_19[(df_q14_19["Shortage"] == 2)])
q16_1 = len(df_q14_19[(df_q14_19["Shortage"] == 1)])
print("[Q16] A5 : {}, A4 : {}, A3 : {}, A2 : {}, A1 : {}".format(
    round(q16_5 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3), 
    round(q16_4 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3),
    round(q16_3 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3),
    round(q16_2 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3),
    round(q16_1 / (q14_5 + q16_4 + q16_3 + q16_2 + q14_1), 3))
)

q17_5 = len(df_q14_19[(df_q14_19["Support"] == 5)])
q17_4 = len(df_q14_19[(df_q14_19["Support"] == 4)])
q17_3 = len(df_q14_19[(df_q14_19["Support"] == 3)])
q17_2 = len(df_q14_19[(df_q14_19["Support"] == 2)])
q17_1 = len(df_q14_19[(df_q14_19["Support"] == 1)])
print("[Q17] A5 : {}, A4 : {}, A3 : {}, A2 : {}, A1 : {}".format(
    round(q17_5 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3), 
    round(q17_4 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3),
    round(q17_3 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3),
    round(q17_2 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3),
    round(q17_1 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3))
)

q18_5 = len(df_q14_19[(df_q14_19["Culture"] == 5)])
q18_4 = len(df_q14_19[(df_q14_19["Culture"] == 4)])
q18_3 = len(df_q14_19[(df_q14_19["Culture"] == 3)])
q18_2 = len(df_q14_19[(df_q14_19["Culture"] == 2)])
q18_1 = len(df_q14_19[(df_q14_19["Culture"] == 1)])
print("[Q18] A5 : {}, A4 : {}, A3 : {}, A2 : {}, A1 : {}".format(
    round(q18_5 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3), 
    round(q18_4 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3),
    round(q18_3 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3),
    round(q18_2 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3),
    round(q18_1 / (q18_5 + q18_4 + q14_3 + q18_2 + q18_1), 3))
)

q19_5 = len(df_q14_19[(df_q14_19["Amongthehindrance"] == 5)])
q19_4 = len(df_q14_19[(df_q14_19["Amongthehindrance"] == 4)])
q19_3 = len(df_q14_19[(df_q14_19["Amongthehindrance"] == 3)])
q19_2 = len(df_q14_19[(df_q14_19["Amongthehindrance"] == 2)])
q19_1 = len(df_q14_19[(df_q14_19["Amongthehindrance"] == 1)])
print("[Q19] A5 : {}, A4 : {}, A3 : {}, A2 : {}, A1 : {}".format(
    round(q19_5 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3), 
    round(q19_4 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3),
    round(q19_3 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3),
    round(q19_2 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3),
    round(q19_1 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3))
)

#%%
index = ['Q14','Q15','Q16','Q17','Q18','Q19']
columns = ['ans5','ans4','ans3','ans2','ans1']

data_ = np.array([[round(q14_5 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3), 
                round(q14_4 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3),
                round(q14_3 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3),
                round(q14_2 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3),
                round(q14_1 / (q14_5 + q14_4 + q14_3 + q14_2 + q14_1), 3)],
                [round(q15_5 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3), 
                round(q15_4 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3),
                round(q15_3 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3),
                round(q15_2 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3),
                round(q15_1 / (q15_5 + q15_4 + q15_3 + q15_2 + q15_1), 3)],
                [round(q16_5 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3), 
                round(q16_4 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3),
                round(q16_3 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3),
                round(q16_2 / (q16_5 + q16_4 + q16_3 + q16_2 + q16_1), 3),
                round(q16_1 / (q14_5 + q16_4 + q16_3 + q16_2 + q14_1), 3)],
                [round(q17_5 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3), 
                round(q17_4 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3),
                round(q17_3 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3),
                round(q17_2 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3),
                round(q17_1 / (q17_5 + q17_4 + q17_3 + q17_2 + q17_1), 3)],
                [round(q18_5 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3), 
                round(q18_4 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3),
                round(q18_3 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3),
                round(q18_2 / (q18_5 + q18_4 + q18_3 + q18_2 + q18_1), 3),
                round(q18_1 / (q18_5 + q18_4 + q14_3 + q18_2 + q18_1), 3)],
                [round(q19_5 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3), 
                round(q19_4 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3),
                round(q19_3 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3),
                round(q19_2 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3),
                round(q19_1 / (q19_5 + q19_4 + q19_3 + q19_2 + q19_1), 3)]
                ])

work2 = pd.DataFrame(data_, index=index, columns=columns)
print(work2)
#%% Work 3
data.columns
data['Sex']
X_ = data.loc[:, ["Sex", "Age", "education", "Income", "Familysize", "Wherewaste"]]
Y_ = data.loc[:, ["Wastemanagement", "Mixedwaste", "Segregation"]]
Y = Y_.copy()
Y["Wastemanagement"] = Y_["Wastemanagement"].map({1:1, 2:0})
Y["Mixedwaste"] = Y_["Mixedwaste"].map({1:1, 2:0})
Y["Segregation"] = Y_["Segregation"].map({1:1, 2:0})

factor_ = data.loc[:, ['importanceofpolicy', 'Individualcontribution',
       'Ifanypolicy', 'Punishmentfornotpracticing', 'Economicencentive']]

# Scree Plot
pca = PCA(n_components = 5)
pc = pca.fit_transform(factor_)

num_components = len(pca.explained_variance_ratio_)
ind = np.arange(num_components)
vals = pca.explained_variance_ratio_
ax = plt.subplot()
cumvals = np.cumsum(vals)
ax.bar(ind, vals, color = ['#00da75', '#f1c40f', '#ff6f15', '#3498db'])
ax.plot(ind, cumvals, color = '#c0302b')

for i in range(num_components):
    ax.annotate(r"%s" %((str(vals[i] * 100)[:3])), (ind[i], vals[i]), va = "bottom", ha = "center", fontsize = 13)
    
ax.set_xlabel("PC")
ax.set_ylabel("variance")
plt.title("Scree plot")

# FA
fa = FactorAnalysis(rotation = 'varimax')
fa.set_params(n_components = 3)
fa.fit(factor_)

variables = fa.fit_transform(factor_)
col_tmp = ['factor1','factor2', 'factor3']
factor = pd.DataFrame(variables, index=None, columns = col_tmp)

work3_factor = factor
# concat
X_ = pd.concat([X_, factor], axis=1)

#%%
X = sm.add_constant(X_)
res1 = sm.Logit(Y.iloc[:, 0:1], sm.add_constant(X)).fit()
res2 = sm.Logit(Y.iloc[:, 1:2], sm.add_constant(X)).fit()
res3 = sm.Logit(Y.iloc[:, 2:3], sm.add_constant(X)).fit()

work3_y1_all = res1.summary()

work3_y2_all = res2.summary()

work3_y3_all = res3.summary()

#%%work3

#%% forward selection

variables = X_.columns ## ?????? ?????? ?????????

selected_variables = [] ## ????????? ?????????
sl_enter = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(variables)
while iter > 0:
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) ## ????????? p-value
    ## ????????? ????????? ????????? ????????? ?????? ????????? ??????????????? 
    ## ?????? ????????? ????????????.
    for col in remainder: 
        X = X_[selected_variables+[col]]
        X = sm.add_constant(X)
        model = sm.Logit(Y.iloc[:, 0:1],X).fit()
        pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < sl_enter: ## ?????? p-value ?????? ?????? ????????? ????????? ??????
        selected_variables.append(pval.idxmin())
        
        step += 1
        steps.append(step)
        bic = sm.Logit(Y.iloc[:, 0:1],sm.add_constant(X_[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y.iloc[:, 0:1],sm.add_constant(X_[selected_variables])).fit()
        break
work3_y1_forward = res.summary()
#%% backward elimination

variables = X_.columns

selected_variables = list(set(variables)) ## ???????????? ?????? ????????? ????????? ??????
sl_remove = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(selected_variables)
while iter > 0:
    X = sm.add_constant(X_[selected_variables])
    p_vals = sm.Logit(Y.iloc[:, 0:1], X).fit().pvalues[1:] ## ???????????? p-value??? ??????
    max_pval = p_vals.max() ## ?????? p-value
    if max_pval >= sl_remove: ## ?????? p-value?????? ??????????????? ????????? ????????? ??????
        remove_variable = p_vals.idxmax()
        selected_variables.remove(remove_variable)

        step += 1
        steps.append(step)
        bic = sm.Logit(Y.iloc[:, 0:1],sm.add_constant(X[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y.iloc[:, 0:1],sm.add_constant(X[selected_variables])).fit()
        break

work3_y1_backward = res.summary()
#%% y2 forward selection

variables = X_.columns ## ?????? ?????? ?????????

selected_variables = [] ## ????????? ?????????
sl_enter = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(variables)
while iter > 0:
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) ## ????????? p-value
    ## ????????? ????????? ????????? ????????? ?????? ????????? ??????????????? 
    ## ?????? ????????? ????????????.
    for col in remainder: 
        X = X_[selected_variables+[col]]
        X = sm.add_constant(X)
        model = sm.Logit(Y.iloc[:, 1:2],X).fit()
        pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < sl_enter: ## ?????? p-value ?????? ?????? ????????? ????????? ??????
        selected_variables.append(pval.idxmin())
        
        step += 1
        steps.append(step)
        bic = sm.Logit(Y.iloc[:, 1:2],sm.add_constant(X_[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y.iloc[:, 1:2],sm.add_constant(X_[selected_variables])).fit()
        break
work3_y2_forward = res.summary()
#%% backward elimination

variables = X_.columns

selected_variables = list(set(variables)) ## ???????????? ?????? ????????? ????????? ??????
sl_remove = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(selected_variables)
while iter > 0:
    X = sm.add_constant(X_[selected_variables])
    p_vals = sm.Logit(Y.iloc[:, 1:2], X).fit().pvalues[1:] ## ???????????? p-value??? ??????
    max_pval = p_vals.max() ## ?????? p-value
    if max_pval >= sl_remove: ## ?????? p-value?????? ??????????????? ????????? ????????? ??????
        remove_variable = p_vals.idxmax()
        selected_variables.remove(remove_variable)

        step += 1
        steps.append(step)
        bic = sm.Logit(Y.iloc[:, 1:2],sm.add_constant(X[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y.iloc[:, 1:2],sm.add_constant(X[selected_variables])).fit()
        break

work3_y2_backward = res.summary()
#%% forward selection

variables = X_.columns ## ?????? ?????? ?????????

selected_variables = [] ## ????????? ?????????
sl_enter = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(variables)
while iter > 0:
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) ## ????????? p-value
    ## ????????? ????????? ????????? ????????? ?????? ????????? ??????????????? 
    ## ?????? ????????? ????????????.
    for col in remainder: 
        X = X_[selected_variables+[col]]
        X = sm.add_constant(X)
        model = sm.Logit(Y.iloc[:, 2:3],X).fit()
        pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < sl_enter: ## ?????? p-value ?????? ?????? ????????? ????????? ??????
        selected_variables.append(pval.idxmin())
        
        step += 1
        steps.append(step)
        bic = sm.Logit(Y.iloc[:, 2:3],sm.add_constant(X_[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y.iloc[:, 2:3],sm.add_constant(X_[selected_variables])).fit()
        break
work3_y3_forward = res.summary()
#%% backward elimination

variables = X_.columns

selected_variables = list(set(variables)) ## ???????????? ?????? ????????? ????????? ??????
sl_remove = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(selected_variables)
while iter > 0:
    X = sm.add_constant(X_[selected_variables])
    p_vals = sm.Logit(Y.iloc[:, 2:3], X).fit().pvalues[1:] ## ???????????? p-value??? ??????
    max_pval = p_vals.max() ## ?????? p-value
    if max_pval >= sl_remove: ## ?????? p-value?????? ??????????????? ????????? ????????? ??????
        remove_variable = p_vals.idxmax()
        selected_variables.remove(remove_variable)

        step += 1
        steps.append(step)
        bic = sm.Logit(Y.iloc[:, 2:3],sm.add_constant(X[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y.iloc[:, 2:3],sm.add_constant(X[selected_variables])).fit()
        break

work3_y3_backward = res.summary()
#%% Work 5

factor_ = data.loc[:, ['Importanceofsegregation', 'Segregationisthebiggestproblem',
       'Dutyofmunicipality', 'Individualcannotminimizewaste',
       'timeandresourceforsegregation', 'householdmanadate',
       'Harmfultoenvironment', 'importanceofpolicy', 'Individualcontribution',
       'Ifanypolicy', 'Punishmentfornotpracticing', 'Economicencentive']]

# Scree plot
pca = PCA(n_components = factor_.shape[1])
pc = pca.fit_transform(factor_)

num_components = len(pca.explained_variance_ratio_)
ind = np.arange(num_components)
vals = pca.explained_variance_ratio_
ax = plt.subplot()
cumvals = np.cumsum(vals)
ax.bar(ind, vals, color = ['#00da75', '#f1c40f', '#ff6f15', '#3498db'])
ax.plot(ind, cumvals, color = '#c0302b')

for i in range(num_components):
    ax.annotate(r"%s" %((str(vals[i] * 100)[:3])), (ind[i], vals[i]), va = "bottom", ha = "center", fontsize = 13)
    
ax.set_xlabel("PC")
ax.set_ylabel("variance")
plt.title("Scree plot")

# FA
fa = FactorAnalysis(rotation = 'varimax')
fa.set_params(n_components = 5)
fa.fit(factor_)

variables = fa.fit_transform(factor_)
col_tmp = ['factor1','factor2', 'factor3', 'factor4', 'factor5']
factor = pd.DataFrame(variables, index=None, columns = col_tmp)

work5 = factor

#%% Work 4
idx = data[data['Mixedwaste'] == 2].index
data_ = data.drop(idx)
data_ = data_.reset_index()
X_ = data_.loc[:, ["Sex", "Age", "education", "Income", "Familysize", "Wherewaste"]]

Y_ = data_.loc[:, ["Segregation"]]
Y = Y_.copy()
Y["Segregation"] = Y_["Segregation"].map({1:1, 2:0})


factor_ = data_.loc[:, ['importanceofpolicy', 'Individualcontribution',
       'Ifanypolicy', 'Punishmentfornotpracticing', 'Economicencentive']]

# Scree Plot
pca = PCA(n_components = 5)
pc = pca.fit_transform(factor_)

num_components = len(pca.explained_variance_ratio_)
ind = np.arange(num_components)
vals = pca.explained_variance_ratio_
ax = plt.subplot()
cumvals = np.cumsum(vals)
ax.bar(ind, vals, color = ['#00da75', '#f1c40f', '#ff6f15', '#3498db'])
ax.plot(ind, cumvals, color = '#c0302b')

for i in range(num_components):
    ax.annotate(r"%s" %((str(vals[i] * 100)[:3])), (ind[i], vals[i]), va = "bottom", ha = "center", fontsize = 13)
    
ax.set_xlabel("PC")
ax.set_ylabel("variance")
plt.title("Scree plot")

# FA
fa = FactorAnalysis(rotation = 'varimax')
fa.set_params(n_components = 3)
fa.fit(factor_)
variables = fa.fit_transform(factor_)
variables.shape
col_tmp = ['factor1','factor2', 'factor3']
factor = pd.DataFrame(variables, index=None, columns = col_tmp)

# concat
X_ = pd.concat([X_, factor], axis=1)

#%%
X = sm.add_constant(X_)
res = sm.Logit(Y, X).fit()

work4_y3_all = res.summary()

#%% forward selection

variables = X_.columns ## ?????? ?????? ?????????

selected_variables = [] ## ????????? ?????????
sl_enter = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(variables)
while iter > 0:
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) ## ????????? p-value
    ## ????????? ????????? ????????? ????????? ?????? ????????? ??????????????? 
    ## ?????? ????????? ????????????.
    for col in remainder: 
        X = X_[selected_variables+[col]]
        X = sm.add_constant(X)
        model = sm.Logit(Y,X).fit()
        pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < sl_enter: ## ?????? p-value ?????? ?????? ????????? ????????? ??????
        selected_variables.append(pval.idxmin())
        
        step += 1
        steps.append(step)
        bic = sm.Logit(Y,sm.add_constant(X_[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y,sm.add_constant(X_[selected_variables])).fit()
        break
work4_y3_forward = res.summary()
#%% backward elimination

variables = X_.columns

selected_variables = list(set(variables)) ## ???????????? ?????? ????????? ????????? ??????
sl_remove = 0.1

sv_per_step = [] ## ??? ???????????? ????????? ?????????
bic_lst = [] ## ??? ????????? ????????? ????????????
steps = [] ## ??????
step = 0
iter = len(selected_variables)
while iter > 0:
    X = sm.add_constant(X_[selected_variables])
    p_vals = sm.Logit(Y, X).fit().pvalues[1:] ## ???????????? p-value??? ??????
    max_pval = p_vals.max() ## ?????? p-value
    if max_pval >= sl_remove: ## ?????? p-value?????? ??????????????? ????????? ????????? ??????
        remove_variable = p_vals.idxmax()
        selected_variables.remove(remove_variable)

        step += 1
        steps.append(step)
        bic = sm.Logit(Y,sm.add_constant(X[selected_variables])).fit().bic
        bic_lst.append(bic)
        sv_per_step.append(selected_variables.copy())
    else:
        iter = 0
        res = sm.Logit(Y,sm.add_constant(X[selected_variables])).fit()
        break

work4_y3_backward = res.summary()
# %%
# os.chdir(r"C:\won\data\seperate")
os.chdir("/Users/wonhyung64/data/seperate")
work = [work1, work2, work3_y1_all, work3_y2_all, 
        work3_y3_all, work3_y1_forward, work3_y1_backward,
        work3_y2_forward, work3_y2_backward, work3_y3_forward,
        work3_y3_backward, work3_factor, work5,
        work4_y3_all, work4_y3_forward, work4_y3_backward]
file_names = ["work1", "work2", "work3_y1_all", "work3_y2_all", 
        "work3_y3_all", "work3_y1_forward", "work3_y1_backward",
        "work3_y2_forward", "work3_y2_backward", "work3_y3_forward",
        "work3_y3_backward", "work3_factor", "work5",
        "work4_y3_all", "work4_y3_forward", "work4_y3_backward"]
for i in range(len(file_names)):
    try:
        with open(file_names[i]+'.csv', 'w')  as f:
            f.write(work[i].to_csv())
    except:
        with open(file_names[i]+'.csv', 'w')  as f:
            f.write(work[i].as_csv())
# %%

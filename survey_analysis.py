#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA

# %%
os.chdir("E:\Data\seperate_garbage")
data = pd.read_excel("House hold solid waste segregation  data.xls")

data.columns
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

# concat
X_ = pd.concat([X_, factor], axis=1)

#%%
import statsmodels.api as sm
from sklearn.datasets import make_blobs

res1 = sm.Logit(Y.iloc[:, 0:1], sm.add_constant(X)).fit()
res2 = sm.Logit(Y.iloc[:, 1:2], sm.add_constant(X)).fit()
res3 = sm.Logit(Y.iloc[:, 2:3], sm.add_constant(X)).fit()

print(res1.summary())
print(res2.summary())
print(res3.summary())

#%% forward selection

variables = X_.columns ## 설명 변수 리스트

selected_variables = [] ## 선택된 변수들
sl_enter = 0.05

sv_per_step = [] ## 각 스텝별로 선택된 변수들
adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
steps = [] ## 스텝
step = 0
while len(variables) > 0:
    remainder = list(set(variables) - set(selected_variables))
    pval = pd.Series(index=remainder) ## 변수의 p-value
    ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서 
    ## 선형 모형을 적합한다.
    for col in remainder: 
        X = X_[selected_variables+[col]]
        X = sm.add_constant(X)
        model = sm.Logit(Y.iloc[:, 0:1],X).fit()
        pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < sl_enter: ## 최소 p-value 값이 기준 값보다 작으면 포함
        selected_variables.append(pval.idxmin())
        
        step += 1
        steps.append(step)
        adj_r_squared = sm.Logit(Y.iloc[:, 0:1],sm.add_constant(X[selected_variables])).fit().rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected_variables.copy())
    else:
        break

#%% backward elimination

variables = X_.columns

selected_variables = variables ## 초기에는 모든 변수가 선택된 상태
sl_remove = 0.05

sv_per_step = [] ## 각 스텝별로 선택된 변수들
adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
steps = [] ## 스텝
step = 0
while len(selected_variables) > 0:
    X = sm.add_constant(X_[selected_variables])
    p_vals = sm.Logit(Y.iloc[:, 0:1], X).fit().pvalues[1:] ## 절편항의 p-value는 뺀다
    max_pval = p_vals.max() ## 최대 p-value
    if max_pval >= sl_remove: ## 최대 p-value값이 기준값보다 크거나 같으면 제외
        remove_variable = p_vals.idxmax()
        selected_variables.remove(remove_variable)

        step += 1
        steps.append(step)
        adj_r_squared = sm.Logit(Y.iloc[:, 0:1],sm.add_constant(X[selected_variables])).fit().rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected_variables.copy())
    else:
        break

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

# %%

#%%
import os
import pandas as pd


# %%
os.chdir("E:\Data\seperate_garbage")
data = pd.read_excel("House hold solid waste segregation  data.xls")

data.columns
# %% Work 1
df_q7 = data["Mixedwaste"]
df_q6 = data["Wastemanagement"]
df_q11 = data["Segregation"]
#%% Work 2
df_q14_19 = data.iloc[:,13:19]

#%% Work 3

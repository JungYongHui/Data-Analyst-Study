# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:11:09 2022

@author: sjnam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/해외 축구 선수 이적료 예측/FIFA_train.csv')
test = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/해외 축구 선수 이적료 예측/FIFA_test.csv')
submission = pd.read_csv('C:/Users/sjnam/Desktop/데이콘 데이터/해외 축구 선수 이적료 예측/submission.csv')

train.info()
train.isnull().sum()
train.describe()
train.describe(include='object')

nums = ['age', 'stat_overall', 'stat_potential'] # 연속형 (continuous)
noms = ['continent', 'contract_until' ,'position', 'prefer_foot', 'reputation', 'stat_skill_moves'] # 이산형 (nominal)
y = 'value'

# 이산형 변수의 개수 확인
for col in ['continent', 'position', 'prefer_foot', 'contract_until','reputation', 'stat_skill_moves']:
  print(col)
  print(train[col].value_counts())

f, axes = plt.subplots(3,4, figsize=(30,18))
axes = axes.flatten()
for col, ax in zip(train.columns, axes):
  sns.histplot(data = train, x=col, ax=ax)
plt.show()
# 아이디와 이름은 삭제해도 괜찮을거 같음, 유럽선수가 많다

#연속형 자료 시각화

f, axes = plt.subplots(1,len(nums), figsize=(24,5))
axes = axes.flatten()                         
for col, ax in zip(nums, axes):
  sns.histplot(data = train, x=col, ax=ax) # 연속형 자료는 histplot 사용
plt.show()
# --> 자료가 고르게 분포 되어있음


# 이산형 자료 시각화

f, axes = plt.subplots(1,len(noms), figsize=(28,5))
axes = axes.flatten()                         
for col, ax in zip(noms, axes):
  sns.countplot(data = train, x=col, ax=ax) # 이산형은 countplot 사용
plt.show()
# --> 해석 해보기

# 연속형 자료 시각화 (boxplot)

f, axes = plt.subplots(1,len(nums), figsize=(24,5))
axes = axes.flatten()                         
for col, ax in zip(nums, axes):
  sns.boxplot(data = train, x=col, ax=ax)
plt.show()

sns.barplot(data = train, x='position', y=y) # 포지션과 몸값 사이의 관계
sns.barplot(data = train, x='prefer_foot', y=y) # 주로쓰는 발과 몸값 사이의 관계
sns.heatmap(data= train.corr(), cmap='coolwarm', annot=True, vmax=1, vmin=-1)


# 범주형 변수랑 예측값과의 상관관계

f, axes = plt.subplots(1,len(noms), figsize=(30,5))
axes = axes.flatten()                         
for col, ax in zip(noms, axes):
  sns.boxplot(data = train, x=col, y=y,ax=ax) # 박스플롯을 사용
plt.show()

# 연속형 변수랑 예측값과의 상관관계
f, axes = plt.subplots(1,len(nums), figsize=(20,5))
axes = axes.flatten()                         
for col, ax in zip(nums, axes):
  sns.scatterplot(data = train, x=col, y=y, ax=ax, hue = 'continent') # 대륙별로 색을 지정
plt.show()

# 대륙 따로따로 나타내기
for col, ax in zip(nums, axes):
  sns.lmplot(data = train, x=col, y=y, hue='continent', col='continent') # 대륙별로 위 그래프를 분리
  plt.show()
  
# 선호하는 발마다 나타내기
for col, ax in zip(nums, axes):
  sns.lmplot(data = train, x=col, y=y, hue='prefer_foot', col='prefer_foot') # 선호하는 발로 그래프를 분리
  plt.show()
  
# 2주차 시작

f, axes = plt.subplots(1,3)
axes = axes.flatten()
# 이적료에 log변환
train["log_value"] = np.log(train["value"])
sns.histplot(x = 'value', data = train, ax = axes[0])
axes[0].set(title = '이적료 히스토그램')
sns.histplot(x = 'log_value', data = train, ax = axes[1])
axes[1].set(title = '이적료_로그스케일 히스토그램')
sns.boxplot(x = 'log_value', data = train, ax = axes[2])
axes[2].set(title = '이적료_로그스케일 박스플롯')

# 이적료에 이상치가 많아서 로그 스케일로 정규화 하는 방향으로 감 
# 로그 스케일임에도 불구하고 이상치가 있음 10 ~ 18까지만 사용해도 괜찮을듯

f.ax = plt.subplot()

group_age_value = train.groupby("age")["value"].mean()
group_age_value = group_age_value.reset_index()

sns.barplot(x = "age", y = 'value', data = group_age_value)

# 나이가 적은 사람과 나이가 많은 사람은 이적료 차이가 너무 많이 남 -> 로그 스케일로 변환

group_age_log_value = train.groupby("age")["log_value"].mean()
group_age_log_value = group_age_log_value.reset_index()

sns.barplot(x = "age", y = "log_value", data = group_age_log_value)

# 로그스케일 변환하니까 극단치가 많이 사라짐

f, axes = plt.subplots(1,2)
axes = axes.flatten()

print("stat_overall의 범위 :",np.ptp(train["stat_overall"]))

sns.histplot(x = "stat_overall",data = train, ax = axes[0])
sns.barplot(x = "stat_overall", y = "log_value" ,data = train, ax = axes[1])

# 당연하게도 현재능력치가 높을수록 value가 높음

f, axes = plt.subplots(1,2)
axes = axes.flatten()

print('stat_potential의 범위 :', np.ptp(train["stat_potential"]))

sns.histplot(x = "stat_potential", data = train, ax = axes[0])
sns.barplot(x = "stat_potential", y = "log_value", data = train, ax = axes[1])

# 잠재능력치가 높을수록 value가 높음

train["stat_ap"] = (train["stat_overall"] + train['stat_potential'])/2
train['stat_ap'] = train["stat_ap"].astype(int)

f,axes = plt.subplots(1,2)
axes = axes.flatten()
sns.histplot(x = "stat_ap", data = train, ax = axes[0])
sns.barplot(x = "stat_ap", y = "log_value", data = train, ax = axes[1])

# 현재 능력치랑 잠재 능력치의 평균으로 새로운 변수 생성

group_stat_overall_value = train.groupby("stat_overall")['value'].mean()
group_stat_overall_value = group_stat_overall_value.reset_index()
group_stat_potential_value = train.groupby("stat_potential")['value'].mean()
group_stat_potential_value = group_stat_potential_value.reset_index()
group_stat_ap_value = train.groupby("stat_ap")["value"].mean()
group_stat_ap_value = group_stat_ap_value.reset_index()

group_stat_overall_value.columns = ["stat", 'ov_value']
group_stat_potential_value.columns = ['stat', 'po_value']
group_stat_ap_value.columns = ["stat", 'ap_value']

corrdf = group_stat_ap_value.merge(group_stat_potential_value, how = 'inner', on = 'stat').merge(group_stat_overall_value, how = 'inner', on = 'stat')
corrdf = corrdf.drop('stat', axis = 1)
sns.heatmap(data = corrdf.corr(), annot = True, cmap = 'Blues')

# 셋다 강한 상관관계를 가지기 때문에 현재 능력치와 잠재 능력치 대신에 총 능력치를 사용해도 된다.

f,axes = plt.subplots(1,3)
axes = axes.flatten()

sns.histplot(x = 'continent', data = train, ax = axes[0])
axes[0].tick_params(axis = 'x', labelrotation = 30)

sns.barplot(x = 'continent', y = 'value', data = train, ax = axes[1])
axes[1].tick_params(axis = 'x', labelrotation = 30)

sns.barplot(x = 'continent', y = 'log_value', data = train, ax = axes[2])
axes[2].tick_params(axis = 'x', labelrotation= 30)

# 특이사항이 없음 -> 그냥 인코딩 해도 될듯

train["contract_until"].unique()
train.loc[train['contract_until']=="Jun 30, 2019",'contract_until'] = 2018
train.loc[train['contract_until']=="Dec 31, 2018",'contract_until'] = 2018
train.loc[train['contract_until']=="May 31, 2019",'contract_until'] = 2019
train.loc[train['contract_until']=="Jan 31, 2019",'contract_until'] = 2018
train.loc[train['contract_until']=="Jun 30, 2020",'contract_until'] = 2020
train.loc[train['contract_until']=="Jan 1, 2019",'contract_until'] = 2018
train.loc[train['contract_until']=="May 31, 2020",'contract_until'] = 2020
train.loc[train['contract_until']=="Jan 12, 2019",'contract_until'] = 2018

train['contract_until'] = train['contract_until'].astype(int)

# 1월에 계약이 끝나는 선수들은 그 전년도로 처리함

f, axes = plt.subplots(1,3)
axes = axes.flatten()

sns.histplot(x = 'contract_until', data = train, ax = axes[0])
axes[0].tick_params(axis = 'x', labelrotation = 45) 

sns.barplot(x = 'contract_until', y = 'value', data = train, ax = axes[1])
axes[1].tick_params(axis = 'x', labelrotation = 45)

sns.barplot(x = 'contract_until', y = 'log_value', data = train, ax = axes[2])
axes[2].tick_params(axis = 'x', labelrotation = 45)

# 4자리 숫자는 너무 기니까 계약만료까지 남은 기간으로 바꾸면 좋을 듯 (0~8)

train['contract_until'] = train['contract_until'].apply(lambda x: x - 2018 )









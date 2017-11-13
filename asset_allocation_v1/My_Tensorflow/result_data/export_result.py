import pandas as pd
import numpy as np

filepath = 'train_result_1.npz'
res = np.load(filepath)

df_train = pd.concat([pd.DataFrame(res['train_data']), pd.DataFrame(res['train_result'])], 1)
df_train.columns = ['train_data', 'train_result']
df_train['train_result_smooth'] = df_train['train_result'].rolling(30).mean()
df_train.to_csv('train_result.csv')

df_test = pd.concat([pd.DataFrame(res['test_data']), pd.DataFrame(res['test_result'])], 1)
df_test.columns = ['test_data', 'test_result']
df_test['test_result_smooth'] = df_test['test_result'].rolling(30).mean()
df_test.to_csv('test_result.csv')

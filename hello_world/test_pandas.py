import pandas as pd
import numpy as np
import os

def get_app_info_df(xlsx_name):
    col_names=['id','pkg_name','category','name','version','score',
        'cp','cp_model','out_rec','inner_rec','price']
    app_info_pd_df=pd.read_excel(xlsx_name, names=col_names, sheetname='Sheet1')
    for col in col_names:
        app_info_pd_df[col]=app_info_pd_df[col].astype(str)
    app_info_pd_df['index']=range(133)
    app_info_pd_df=app_info_pd_df.replace(np.NaN, '')
    return app_info_pd_df

def get_df(xlsx_name, cat_list, names=['tid', 'title']):
    dfs = []
    for cat in cat_list:
        df = pd.read_excel(xlsx_name, names=names, sheetname=cat)
        df['cat'] = cat
        dfs.append(df)
    all_df = dfs[0]
    for df in dfs[1:]:
        all_df = all_df.append(df)
    return all_df
'''
cat_list = ['general', 'fix', 'student', 'calendar', 'qimao']
xlsx_file_path=os.path.join(os.getcwd(),'career.xlsx')
title_df = get_df(xlsx_file_path, cat_list)
print(title_df)
'''
xlsx2_file_path=os.path.join(os.getcwd(),'app_info_list.xlsx')
app_info_df = get_app_info_df(xlsx2_file_path)
print(app_info_df)
#print(app_info_df['version'])




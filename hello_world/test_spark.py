#coding=utf-8
'''
@author: cuiy
@contact: cuiy@2345.com
@file: generate_user_ad_rec_table.py
@time: 2018/10/30 17:00
@desc: test spark.
'''

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import argparse
from datetime import datetime,timedelta
import os
from pyspark.sql import functions as F
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default='mlg', help='database name')

    args = parser.parse_args()
    return args

def test_query_where_value_in_table(spark_session):
    print("-------test_query_where_value_in_table-------")
    dt_list=[("2018-07-29",),("2018-07-28",),("2018-07-27",)]
    src_data_df=spark_session.createDataFrame(dt_list,schema=["dt"])
    print("list count:",src_data_df.count())
    tmp_data_view='tmp_data_list'
    src_data_df.createOrReplaceTempView("tmp_data_list")
    table_name = f'test_user_records_table'
    #query_table_sql=f'select {table_name}.* from {table_name},{tmp_data_view} where {table_name}.dt={tmp_data_view}.dt'
    join_table_sql=f'select {table_name}.* from  {table_name} inner join {tmp_data_view} on {table_name}.dt={tmp_data_view}.dt'
    result_df=spark_session.sql(join_table_sql)
    print("result count:",result_df.count())


def test_insert_data_into_table(spark_session):
    print("-------test_insert_data_into_table-------")
    dt_list=[(1,2,3),(4,5,6),(7,8,9)]
    src_data_df=spark_session.createDataFrame(dt_list, schema=['a','b','c'])
    tmp_data_view='tmp_user_rec_data'
    src_data_df.createOrReplaceTempView("tmp_user_rec_data")
    table_name = f'test_cuiy'
    delete_table_sql=f'drop table if exists {table_name}'
    create_table_sql=f'create table {table_name} from {tmp_data_view}'
    spark_session.sql(delete_table_sql)
    spark_session.sql(create_table_sql)

def test_query_large_data(spark_session):
    print("-------test_query_large_data-------")
    df=spark_session.sql("select * from tb_browser_user_action")
    print("tb_browser_user_action count:",df.count())
    rows=df.collect()

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass
    return False

def is_item_id(row):
    item_id=row.gid
    if not is_number(item_id):
        item_id=""
    return (row.uid,item_id)

def union_user_dataframe(user_click_df,user_browser_df):
    if user_click_df.count()<1:
        if user_browser_df is None:
            return None
        else:
            return user_browser_df.filter(user_browser_df.gid!="")
    else:
        if user_browser_df is None:
            return user_click_df.filter(user_click_df.gid!="")
        else:
            user_item_df=user_click_df.union(user_browser_df)
            return user_item_df.filter(user_item_df.gid!="")

def test_union_and_filter_dataframe(spark_session):
    dt_list1=[("1","2"),("4","a"),("7",""),("7","9")]
    dt_list2=[("1","2"),("4","5"),("7","8")]
    src_data_df1=spark_session.createDataFrame(dt_list1, schema=['uid','gid'])
    src_data_df2=spark_session.createDataFrame(dt_list2, schema=['uid','gid'])
    print("src_data_df1 count 1:",src_data_df1.count())
    src_data_df1=src_data_df1.rdd.map(lambda x:is_item_id(x)).toDF(['uid','gid'])
    print("src_data_df1 count 2:",src_data_df1.count())
    src_data_df1=union_user_dataframe(src_data_df1,src_data_df2)
    print("src_data_df1 count 3:",src_data_df1.count())
    src_data_df1=src_data_df1.distinct()
    print("src_data_df1 count 4:",src_data_df1.count())
    uid_df=src_data_df1.select("uid").distinct()
    print("uid_df count 5:",uid_df.count())
    print("uid_df cols:",uid_df.columns)

def  test_join_dataframe(spark_session):
    dt_list1=[("1","2"),("4","a"),("7",""),("7","9")]
    dt_list2=[("1","2"),("4","5"),("7","8"),("8","9")]
    src_data_df1=spark_session.createDataFrame(dt_list1, schema=['uid','gid'])
    src_data_df2=spark_session.createDataFrame(dt_list2, schema=['uid','gid'])
    new_df=src_data_df1.join(src_data_df2,src_data_df1.uid==src_data_df2.uid).select(src_data_df1.gid,src_data_df2.gid)
    print("1",new_df.collect())
    new_df=src_data_df1.join(src_data_df2,src_data_df1.uid==src_data_df2.uid,'inner').select(src_data_df1.gid,src_data_df2.gid)
    print("2",new_df.collect())
    new_df=src_data_df1.join(src_data_df2,src_data_df1.uid==src_data_df2.uid,'outer').select(src_data_df1.gid,src_data_df2.gid)
    print("3",new_df.collect())

def joint_string(row):
    new_str=""
    for gid in row.gid:
        new_str+=gid+","
    return(row.uid,new_str)

def test_collect_list_dataframe(spark_session):
    dt_list=[("1","2"),("1","3"),("1","6"),("4","5"),("7","8"),("8","9")]
    src_data_df=spark_session.createDataFrame(dt_list, schema=['uid','gid'])
    new_df=src_data_df.groupby(src_data_df.uid).agg(F.collect_list("gid").alias("gid")).select("uid","gid")
    print(new_df.collect())
    result_df=new_df.rdd.map(lambda row:joint_string(row)).toDF(["uid","gid_list"])
    print(result_df.collect())

def filter_grouped_row(row,product_divisor=8,min_product_num=30):
    product_num=len(row.volume)
    percent_num=int(product_num/product_divisor)
    product_limit_num=product_num
    if product_num>min_product_num:
        if percent_num>min_product_num:
            product_limit_num=percent_num
        else:
            product_limit_num=min_product_num
    sorted_args=np.argsort(-np.array(row.volume))
    for idx in range(product_num):
        if idx>=product_limit_num:
            break
        yield (row.cat_name,row.cat_leaf_name,row.item_id[sorted_args[idx]],row.pic_url[sorted_args[idx]],row.volume[sorted_args[idx]])

def get_ad_info_df(spark_session):
    product_list_sql=f'select item_id,pic_url,volume,cat_name,cat_leaf_name \
        from ad_info where volume>0'
    current_time=datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    print(current_time,product_list_sql)
    product_df=spark_session.sql(product_list_sql)
    current_time=datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    print(current_time,"products count:",product_df.count())
    ordered_df=product_df.orderBy(product_df.volume.desc())
    current_time=datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    print(current_time,"order by volume")
    grouped_df=ordered_df.groupby(product_df.cat_name,product_df.cat_leaf_name)\
        .agg(F.collect_list("volume").alias("volume"),F.collect_list("pic_url").alias("pic_url"),\
        F.collect_list("item_id").alias("item_id"))
    current_time=datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    print(current_time,"group count:",grouped_df.count())
    result_df=grouped_df.rdd.flatMap(lambda row:filter_grouped_row(row,8,30)).\
        toDF(["cat_name","cat_leaf_name","item_id","pic_url","volume"])
    current_time=datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    print(current_time,"result count",result_df.count())
    return result_df

def write_table(spark_session,ad_info_df):
    table_name = f'filtered_ad_info_new_1'
    delete_sql = f'drop table if exists {table_name}'
    tmp_view="ad_info_tmp_view"
    ad_info_df.createOrReplaceTempView(tmp_view)
    create_sql = f"create table {table_name} from {tmp_view}"
    spark_session.sql(delete_sql)
    print(create_sql)
    spark_session.sql(create_sql)

def main(args):
    spark_session = SparkSession.builder \
        .appName("test_spark") \
        .enableHiveSupport() \
        .getOrCreate()
    spark_session.sql(f'use {args.db}')
    write_table(spark_session,get_ad_info_df(spark_session))
    #测试遍历dataframe
    #test_collect_list_dataframe(spark_session)
    #test_union_and_filter_dataframe(spark_session)
    #test_join_dataframe(spark_session)
    #test_query_large_data(spark_session)
    # 测试查询数据
    #test_query_where_value_in_table(spark_session)
    # 测试向表中插入数据
    #test_insert_data_into_table(spark_session)

if __name__ == '__main__':
    args = parse_args()
    main(args)
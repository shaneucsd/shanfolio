import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from utils.file_utils import read_pickle_with_retry


def create_mysql_url(user, password, host, port, database, charset=None):
    """
    创建MySQL连接URL并返回。

    Args:
        user (str): 数据库用户名
        password (str): 数据库密码
        host (str): 数据库主机地址
        port (str): 数据库端口
        database (str): 数据库名称

    Returns:
        str: MySQL连接URL
    """
    if not charset:
        mysql_url = f'mysql://{user}:{password}@{host}:{port}/{database}'
    elif charset:
        mysql_url = f'mysql://{user}:{password}@{host}:{port}/{database}?charset={charset}'
    else:
        pass
    return mysql_url


def create_sql_server_url(username, password, server, port, database):
    """
    连接到 SQL Server 数据库并返回数据库引擎。

    Args:
        username (str): 数据库用户名
        password (str): 数据库密码
        server (str): 数据库服务器地址
        port (str): 数据库端口号
        database (str): 数据库名称

    Returns:
        str: 连接URL
    """

    # dag上只有18的sql server，本地要改成17，现在也用18了
    # if if_local:
    #     connection_string = (f'mssql+pyodbc://{username}:{password}@{server}:{port}/{database}'
    #                          f'?driver=ODBC+Driver+17+for+SQL+Server')
    # else:
    connection_string = (f'mssql+pyodbc://{username}:{password}@{server}:{port}/{database}'
                         f'?driver=ODBC+Driver+18+for+SQL+Server')
    return connection_string


def execute_sql(sql_file, mysql_url, output_dir):
    # 创建数据库连接， sql server特殊处理
    if 'odbc' in mysql_url:
        engine = create_engine(mysql_url,
                               connect_args={
                                   "TrustServerCertificate": "yes"
                               })
        print(engine)
        print(engine)
        print(engine)
        print(engine)
        print(engine)
    else:
        engine = create_engine(mysql_url)

    # 读取SQL文件内容
    try:
        with open(sql_file, 'r', encoding='utf-8') as file:
            sql_content = file.read()
    except:
        with open(sql_file, 'r', encoding='gbk') as file:
            sql_content = file.read()

    sql_content = sql_content.format('19000101')
    sql_content = text(sql_content)

    print(sql_content)

    # 执行SQL语句
    with engine.connect() as connection:
        result = connection.execute(sql_content)

        # sql server的fetchall必须在connection关之前执行，mysql不用
        # 因此这里必须缩进

        # 将查询结果转换为DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        # !!!!!!!!!!!!!! 数据到这里都变成了object类型 !!!!!
        print(df.dtypes)
        output_file = output_dir / (sql_file.stem + '.pickle')

        # 保存为pickle文件
        df.to_pickle(output_file)
    return df


recompute_days = 30


def execute_sql_incremental(sql_file, mysql_url, output_dir):
    # 创建数据库连接， sql server特殊处理
    if 'odbc' in mysql_url:
        engine = create_engine(mysql_url,
                               connect_args={
                                   "TrustServerCertificate": "yes"
                               })
    else:
        engine = create_engine(mysql_url)

    # 读取SQL文件内容
    try:
        with open(sql_file, 'r', encoding='utf-8') as file:
            sql_content = file.read()
    except:
        with open(sql_file, 'r', encoding='gbk') as file:
            sql_content = file.read()

    # Get the previous query date
    # last_date 肯定不能为空，在我们的设计中。如果要跑全量，请看ipynb
    last_date = get_last_date_from_filename(Path(sql_file), output_dir)

    print(f'{sql_file.stem}: last updated at {last_date}, updating..., file name {sql_file.stem}')

    # 回推n日 读取sql，更新数据
    # Specify the date of sql, leave a 20 days delay for data updates.
    last_date_for_sql = (pd.to_datetime(last_date) - pd.Timedelta(days=recompute_days)).strftime('%Y%m%d')
    sql_content = sql_content.format(last_date_for_sql)
    sql_content = text(sql_content)

    # Read SQL
    with engine.connect() as connection:
        result = connection.execute(sql_content)
        df_new = pd.DataFrame(result.fetchall(), columns=result.keys())

    print(f'{sql_file.stem}: fetched, concating data ... , Length of df_new {len(df_new)}')
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = Path(output_dir) / f"{Path(sql_file).stem}_{timestamp}.pickle"

    last_output_file = Path(output_dir) / f"{Path(sql_file).stem}_{last_date}.pickle"
    print('last out put file : ', last_output_file)
    df_old = read_pickle_with_retry(last_output_file)
    df = check_and_correct_data(df_old, df_new)
    try:
        os.remove(last_output_file)
    except:
        print(f' Failed to Remove {last_output_file}')
    df.to_pickle(output_file)
    print(f'{sql_file.stem}: saved ...')


def get_last_date_from_filename(sql_file, output_dir):
    previous_files = [f for f in Path(output_dir).iterdir() if
                      f.suffix == '.pickle' and f.stem.startswith(sql_file.stem)]
    if not previous_files:
        raise FileNotFoundError(
            f'找不到满足日期格式的历史数据文件；{sql_file.stem} airflow上跑不了全量SQL，建议本地运行后上传')
    latest_file = max(previous_files, key=os.path.getctime)
    date_match = re.search(r"_(\d{8})", latest_file.stem)
    if date_match:
        return date_match.group(1)
    else:
        return None


def check_and_correct_data(df, df_new):
    # Check the date column of the sql file.
    print('Columns of old df : ', df.columns)
    print('Columns of new df : ', df_new.columns)

    date_col = [i for i in df.columns if i in ['tradedate', 'enddate', 'publishdate', 'PUBLISHDATE']][0]
    update_start_date = pd.to_datetime(df_new[date_col]).min()
    old_data = df[pd.to_datetime(df[date_col]) < update_start_date]
    new_data = df_new[pd.to_datetime(df_new[date_col]) >= update_start_date]
    df = pd.concat([old_data, new_data])
    df = df.drop_duplicates(subset=['symbol', date_col], keep='last')
    print('Old data: starts from ', pd.to_datetime(old_data[date_col]).min(), ' - ',
          pd.to_datetime(old_data[date_col]).max())
    print('Updated data: starts from ', pd.to_datetime(new_data[date_col]).min(), ' - ',
          pd.to_datetime(new_data[date_col]).max())
    print('Concated : starts from ', pd.to_datetime(df[date_col]).min(), ' - ', pd.to_datetime(df[date_col]).max())

    return df

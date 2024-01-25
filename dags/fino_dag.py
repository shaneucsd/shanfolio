from datetime import datetime

import requests
from airflow.decorators import dag, task


# 创建AirFlow Dag
@dag(dag_id='fino', start_date=datetime(2023, 11, 27), schedule_interval='30 8 * * *', catchup=False,
     tags=['test'])
def fino_dag():
    @task(task_id='main_asset_price')
    def main_asset_price():
        url = 'http://192.168.1.98:60022/indicator'
        headers = {'Content-Type': 'application/json'}
        response = requests.put(url, headers)
        if response.status_code == 200:
            print('success')
        else:
            raise AssertionError("失敗了")

    main_asset_price()

    @task(task_id='abnormal_monitor')
    def abnormal_monitor():
        url = 'http://192.168.1.98:60022/daily/abnormal_monitor'
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers)
        if response.status_code == 200:
            print('success')
        else:
            raise AssertionError("失敗了")

    abnormal_monitor()


dag = fino_dag()

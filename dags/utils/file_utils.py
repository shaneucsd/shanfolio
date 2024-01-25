import pickle
from pathlib import Path
import pandas as pd
import time

scopes = ('000300', '000905', '000906', '000852', '399303', '000985', 'all')
# 如果你想保存文件，请把变量名和路径名定义在下方map中
# 请保持命名规范，使用变量名.pkl存放
file_path_map = {
    "basic_info_stock": "output/basic_info/basic_info_stock.pkl",
    "basic_info_index_board": "output/basic_info/basic_info_index_board.pkl",
    "basic_info_index_swlevel1": "output/basic_info/basic_info_index_swlevel1.pkl",
    "basic_info_index_swlevel2": "output/basic_info/basic_info_index_swlevel2.pkl",
    "basic_info_index_topic": "output/basic_info/basic_info_index_topic.pkl",
    "tradeassets_stock": "output/trade_calendar/tradeassets_stock.pkl",
    "tradeassets_index_board": "output/trade_calendar/tradeassets_index_board.pkl",
    "tradeassets_index_swlevel1": "output/trade_calendar/tradeassets_index_swlevel1.pkl",
    "tradeassets_index_swlevel2": "output/trade_calendar/tradeassets_index_swlevel2.pkl",
    "tradeassets_index_topic": "output/trade_calendar/tradeassets_index_topic.pkl",
    "tradedays": "output/trade_calendar/tradedays.pkl",
    "trade_calendar": "output/trade_calendar/trade_calendar.pkl",
    "index_components": "output/basic_info/index_components.pkl",
    "stock_close": "output/factor/stock/close.pkl",
    "stock_volume": "output/factor/stock/volume.pkl",
    "stock_amount": "output/factor/stock/amount.pkl",
    "stock_righaggr": "output/factor/stock/righaggr.pkl",
    "index_close": "output/factor/index/close.pkl",
    "size": r"output/processed_factors/size.pkl",
    "liquidity": r"output/processed_factors/liquidity.pkl",
    "stock_pool_000300": "output/trade_calendar/stock_pool_000300.pkl",
    "stock_pool_000905": "output/trade_calendar/stock_pool_000905.pkl",
    "stock_pool_000906": "output/trade_calendar/stock_pool_000906.pkl",
    "stock_pool_000852": "output/trade_calendar/stock_pool_000852.pkl",
    "stock_pool_399303": "output/trade_calendar/stock_pool_399303.pkl",
    "stock_pool_000985": "output/trade_calendar/stock_pool_000985.pkl",
    "stock_pool_all": "output/trade_calendar/stock_pool_all.pkl",
}

SQL_TASKS_PATH = "output/sql_tasks"
FACTOR_PATH = "output/factor/"
PROCESSED_FACTOR_PATH = "output/processed_factors/"


def find_airflow_directory(start_path="."):
    """
    Find the "airflow" directory starting from the given path.
    Traverse up the directory tree until it finds the "airflow" directory.
    """
    current_path = Path(start_path).resolve()
    try:
        while True:
            if "airflow" in [p.name for p in current_path.iterdir()]:
                return current_path / "airflow"
            # Move to parent directory
            new_path = current_path.parent
            if new_path == current_path:
                raise FileNotFoundError("Could not find 'airflow' directory.")
            current_path = new_path
    except:
        current_path = Path('/Users/apple/Documents/Pycharm/QushiDataPlatform/airflow')

    return current_path


# Locate the "airflow" directory
airflow_directory = find_airflow_directory()


# Update the file paths
file_path_map = {
    k: (airflow_directory / v).as_posix() for k, v in file_path_map.items()
}
SQL_TASKS_PATH = (airflow_directory / SQL_TASKS_PATH).as_posix()
FACTOR_PATH = (airflow_directory / FACTOR_PATH).as_posix()
PROCESSED_FACTOR_PATH = (airflow_directory / PROCESSED_FACTOR_PATH).as_posix()


def save_object_to_pickle(obj, tag):
    """
    将对象保存为 pickle 文件，如果目录不存在则创建目录。

    Parameters:
        obj (any): 要保存的对象。
        tag: 变量名

    Returns:
        None
    """
    file_path = file_path_map.get(tag)
    if file_path is None:
        raise FileNotFoundError("需要提前在 file_path_map 中定义存放的对象和路径")
    # 将文件路径转换为 Path 对象
    file_path = Path(file_path)

    # 获取目录部分的路径
    directory = file_path.parent

    # 检查目录是否存在，如果不存在则创建
    directory.mkdir(parents=True, exist_ok=True)

    # 将对象保存为 pickle 文件
    with file_path.open("wb") as file:
        pickle.dump(obj, file)

    # print(f"Object saved to {file_path}")


# save_object_to_pickle_with_path，那么就用load_object_from_pickle_with_path来读
def save_object_to_pickle_with_path(obj, tag, directory):
    file_path = Path(directory) / tag

    # 获取目录部分的路径
    directory = file_path.parent

    # 检查目录是否存在，如果不存在则创建
    directory.mkdir(parents=True, exist_ok=True)

    # 将对象保存为 pickle 文件
    with file_path.open("wb") as file:
        pickle.dump(obj, file)

    print(f"{tag} saved to {file_path}")


def load_object_from_pickle_with_path(tag, directory):
    file_path = Path(directory) / tag

    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # 从 pickle 文件加载对象
    with file_path.open("rb") as file:
        loaded_object = pickle.load(file)

    return loaded_object


def load_object_from_pickle(tag):
    """
    从 pickle 文件中加载对象。

    Parameters:
        tag : 变量名。

    Returns:
        any: 从 pickle 文件加载的对象。
    """
    file_path = file_path_map.get(tag)
    if file_path is None:
        raise FileNotFoundError("需要提前在 file_path_map 中定义存放的对象和路径")
    # 将文件路径转换为 Path 对象
    file_path = Path(file_path)

    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    max_retries = 10

    for retry_count in range(max_retries):
        try:
            # 从 pickle 文件加载对象
            with file_path.open("rb") as file:
                loaded_object = pickle.load(file)
            return loaded_object
        except:
            # 如果尚未达到最大重试次数，则等待一段时间后重试
            if retry_count < max_retries - 1:
                wait_time = 5  # 等待时间（秒）
                print(f"Retrying in {wait_time} seconds (Retry {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached ({max_retries}), unable to load pickle file.")
                raise


def read_pickle_from_sql_tasks(tag):
    folder_path = Path(SQL_TASKS_PATH)

    # 遍历文件夹中的文件
    for file_path in folder_path.iterdir():
        # Incremental sql will have date as suffix
        if file_path.stem[:-9] == tag and file_path.suffix == ".pickle":
            # print(f'Read from {folder_path} / {file_path} / {tag}')
            return pd.read_pickle(file_path)
        # Others will not.
        elif file_path.stem == tag and file_path.suffix == ".pickle":
            # print(f'Read from {folder_path} / {file_path} / {tag}')
            return pd.read_pickle(file_path)

    # 如果没有找到匹配的文件，抛出异常
    raise FileNotFoundError(f"没有在output/sql_tasks找到类似的文件: {tag}.pickle，")


def read_pickle_with_retry(file, max_retries=5):
    retry_count = 0
    loaded = False
    while retry_count < max_retries and not loaded:
        try:
            df_old = pd.read_pickle(file)
            loaded = True  # 如果成功加载，设置loaded为True来退出循环
            return df_old
        except (EOFError, FileNotFoundError) as e:
            # 在发生 EOFError 或 FileNotFoundError 时采取特定的处理方式
            if isinstance(e, EOFError):
                print(f"An EOFError occurred: {file}, 文件为空")
            elif isinstance(e, FileNotFoundError):
                print(f"File not found: {file}")

            # 如果尚未达到最大重试次数，则等待一段时间后重试
            if retry_count < max_retries - 1:
                wait_time = 5  # 等待时间（秒）
                print(f"Retrying in {wait_time} seconds (Retry {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                retry_count += 1
            else:
                print(f"Max retries reached ({max_retries}), unable to load pickle file.")
                raise
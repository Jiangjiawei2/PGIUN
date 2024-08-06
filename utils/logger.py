import logging


def get_logger(filename, open_type='a', verbosity=1, name=None):
    """
    生成getLogger对象，用于控制log日志的写入和控制台信息输出
    :param open_type: 打开日志的方式，例如：w， a， w+。。。
    :param filename: 保存日志的路径
    :param verbosity:
    :param name:
    :return:
    """

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

    # 日志输出格式可以自定义，具体每一个代表的内容，可以参阅CSDN的收藏：
    # 输出的日志格式及时间格式
    # log_format = "[%(asctime)s] - [%(filename)s] - [line:%(lineno)d] - [%(levelname)s] ==>  %(message)s"
    log_format = '%(message)s'
    date_format = "%Y/%m/%d %H:%M:%S"
    formatter = logging.Formatter(
        log_format, datefmt=date_format
    )

    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 用于写入日志文件
    fh = logging.FileHandler(filename, open_type)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 用于输出到控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def save_config(config_path, open_type, time, args):
    """
    在指定的保存路径中保存config.txt文件，其中文件中包含的内容为args的具体参数设置
    :param config_path:
    :param open_type: ‘w’，’a‘
    :param time: 用datatime模块来生成
    :param args: 参数设置
    :return: config.txt文本
    """
    with open(config_path, open_type) as f_obj:
        f_obj.write('----------------' + time + '----------------' + '\n\n')
        for arg in vars(args):
            f_obj.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f_obj.write('\n===================================================')
        f_obj.write('\n\n')

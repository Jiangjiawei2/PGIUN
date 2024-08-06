"""
生成分界线，使log的输出更加美观，更容易阅读和辨认
"""


def generate_boundary(log_path, open_type, time):
    with open(log_path, open_type) as f:
        f.write('\n----------------' + time + '----------------\n')

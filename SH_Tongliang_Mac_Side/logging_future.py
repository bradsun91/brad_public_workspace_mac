# coding=utf-8
import logging
import sys
import logging.handlers

# 获取logger实例，空，则返回root logger
logger = logging.getLogger('future')
# 指定输出格式
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s in line %(lineno)d: %(message)s')
# 日志保存到文件
filepath = './error_log/future_error.log'
file_handler = logging.handlers.TimedRotatingFileHandler(
    filepath, when='midnight', interval=1, backupCount=10)
file_handler.setFormatter(formatter)
# 日志输出到控制台
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# 指定输出级别
logger.setLevel(logging.INFO)
"""
调用时，from logging_future import logger
在需要输出的地方logger.info('内容')，
除info外，还有其他方法，如debug, warning, error, critical等
获取并输出Exception时，可用logger.exception('this is an exception: '),
日志中会先出现this is an exception: 接着就是Exception的内容
"""
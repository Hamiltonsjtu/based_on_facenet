#coding: utf-8
import os
import logging 
from logger import handler
LOG_PATH = 'logs'
LOG_FILEBASENAME = 'log'

from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler
# def get_logger_bak(name):
#     logger = logging.getLogger(name)
#     if os.path.exists(LOG_PATH):
#         pass
#     else:
#         os.mkdir(LOG_PATH)
#         
#     LOG_FILE = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
#     # 指定logger输出格式
#     formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
#     # 文件日志
#     #logging.basicConfig(filename="%s/%s" % (LOG_PATH, LOG_FILE), level=logging.DEBUG)
#     
#     file_handler = RotatingFileHandler("%s/%s" % (LOG_PATH, LOG_FILE), mode='a', maxBytes=4*1024*1024, backupCount=10, encoding=None, delay=0)
#     
#     file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
#     # 控制台日志
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.formatter = formatter  # 也可以直接给formatter赋值
#     # 为logger添加的日志处理器，可以自定义日志处理器让其输出到其他地方
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#     # 指定日志的最低输出级别，默认为WARN级别
#     logger.setLevel(logging.ERROR)
#     return logger


def get_filename(filename):
    # Get logs directory
    log_directory = os.path.split(filename)[0]
    print("dir: ", log_directory)
    # Get file extension (also it's a suffix's value (i.e. ".20181231")) without dot
    date = os.path.splitext(filename)[1][1:]
    print("date: ", date)
    # Create new file name
    filename = os.path.join(log_directory, date)
    # I don't want to add index if only one log file will exists for date
    if not os.path.exists('{}.log'.format(filename)):
        return '{}.log'.format(filename)
    # Create new file name with index
    index = 0
    f = '{}.{}.log'.format(filename, index)
    while os.path.exists(f):
        index += 1
        f = '{}.{}.log'.format(filename, index)
    #delete_backup_file(LOG_PATH+'/', 3)
    return f


def get_logger_bak(name):
    if os.path.exists(LOG_PATH):
        pass
    else:
        os.mkdir(LOG_PATH)
    format = u'%(asctime)s\t%(levelname)s\t%(filename)s:%(lineno)d\t%(message)s'
    
#     logger.setLevel(logging.DEBUG)
    # new file every minute
#     rotation_logging_handler = TimedRotatingFileHandler(LOG_PATH+'/'+LOG_FILEBASENAME, 
#                                    when='MIDNIGHT', 
#                                    interval=1, 
#                                    backupCount=3)
    rotation_logging_handler = RotatingFileHandler("%s/%s" % (LOG_PATH, LOG_FILEBASENAME), 
                                                   mode='a', 
                                                   maxBytes=10*1024, # maxBytes=10*1024*1024, #MB
                                                   backupCount=3,  #keep total file number
                                                   encoding=None, 
                                                   delay=0)
    rotation_logging_handler.setLevel(logging.DEBUG)
    rotation_logging_handler.setFormatter(logging.Formatter(format))
    rotation_logging_handler.suffix = '%Y%m%d'
    #rotation_logging_handler.namer = get_filename(LOG_PATH+'/'+LOG_FILEBASENAME)
    logger = logging.getLogger()
    logger.addHandler(rotation_logging_handler)
    return logger

def get_logger(name):
    if not os.path.isdir(LOG_PATH):
        os.makedirs(LOG_PATH)
    log_handler = logging.getLogger(LOG_FILEBASENAME)
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
#     file_handler = RotatingFileHandler(LOG_PATH+'/' +LOG_FILEBASENAME, mode='a', maxBytes=10 * 1024, backupCount=6,
#                                        encoding='utf-8')
    file_handler = handler.MyLoggerHandler(LOG_PATH, LOG_FILEBASENAME, 'D', 20) #base_dir, filename, M-minute, 5: keep 5 files
    file_handler.setFormatter(formatter)
    log_handler.addHandler(file_handler)
    return log_handler 

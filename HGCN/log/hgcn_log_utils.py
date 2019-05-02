import logging


class HGCNLog:
    logger = logging.getLogger('HGCNLog')

    @staticmethod
    def init(model):
        HGCNLog.logger.setLevel(logging.DEBUG)
        filename = './HGCN/log/' + str(model) + '.log'
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(str(model) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        HGCNLog.logger.addHandler(fh)

        logging.basicConfig(level=logging.DEBUG,
                            format=str(model) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S')

    @ staticmethod
    def debug(msg):
        logging.debug(msg)


    @ staticmethod
    def info(msg):
        logging.info(msg)

    @ staticmethod
    def error(msg):
        logging.error(msg)

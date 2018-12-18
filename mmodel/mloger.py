import logging
import datetime
import os
from mmodel.train_capsule import LossBuket
import re

LOG_DIR = ""
FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

class GLOBAL(object):
    _TAG_ = None
    _CHECK_POINT_FLODER_ = './_CKTP/'
    _LOGER_FLODER_ = './_MLOGS/'
    _SUMMARY_LOG_DIR = './SUMMARY.log'

def summary_log():
    
    def __init__():
        logger = logging.getLogger('summary')
        handler = logging.FileHandler(GLOBAL._SUMMARY_LOG_DIR)
        handler = logging.FileHandler(log_file)
        self.record_dic = dict()
        self.record_dic['train'] = list()
        self.record_dic['file'] = list()

    def train(key, value, **kwargs):
        pass
    

def path_helper(base_folder_name, tag_name, create=True):
    '''get a path acordding `base_folder_name` and `tat_name`
    '''

    folder_name = base_folder_name

    # add tag name to path
    if tag_name is not None:
        for i in tag_name.split("/"):
            if(len(i)!=0):
                folder_name += i + '/'

    # add time stamp to folder
    time_folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    folder_name += time_folder_name + "/"
    
    if create:
        if not os.path.exists((folder_name)):
            os.makedirs(folder_name)
    
    return folder_name

def setup_log_folder_name(
    tag_name = None, 
    base_folder_name=GLOBAL._LOGER_FLODER_, 
    create=True
    ):
    '''Set global path to store log file
    
    Keyword Arguments:
        base_folder_name {string} -- the dir for base folder. (default: {None})
        create {bool} -- create if folder not exist. (default: {True})
    '''

    global LOG_DIR
    LOG_DIR = path_helper(base_folder_name=base_folder_name, tag_name=tag_name)
    

class LogCapsule(object):
    
    def __init__(
        self, 
        loss_bucker:LossBuket,
        name, 
        level=logging.INFO,
        to_file=False
        ):
        
    
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if to_file:
            if LOG_DIR is "":
                logging.warn("Current <LOG_DIR is not set>")
                setup_log_folder_name(GLOBAL._TAG_)
                logging.warn("Auto set <LOG_DIR> to <%s>" % LOG_DIR)
            log_file = LOG_DIR + name + '.log'

            handler = logging.FileHandler(log_file)
            handler.setFormatter(FORMATTER)
            logger.addHandler(handler)


        self.LOSS_FORMAT = ">%-12s< at step [%6d] -> [%.3f]."
        self.PRT_FORMAT = "At >%-12s< stage, loss is [%.3f]."

        self.logger = logger
        self.range_loss = None
        self.range_step = 0
        self.loss_bucket = loss_bucker

    def record(self):
        closs = self.loss_bucket.value.item()
        if self.range_loss is None:
            self.range_loss = closs
        else:
            self.range_loss += closs
        self.range_step += 1
    
    def log_current_avg_loss(self, step=None):
        loss = self.avg_record()
        self.__loss__(step, loss)
        return loss

    def avg_record(self):
        try:
            result = self.range_loss / self.range_step
        except:
            result = 0
        self.range_loss = None
        self.range_step = 0
        return result

    def __loss__(self, steps, values):
        '''record loss to log file
        
        Arguments:
            steps {current step} -- steps os current loss
            values {[type]} -- values of current loss
        '''

        if steps is None:
            i = self.PRT_FORMAT % (self.logger.name.upper(), values)
        else:
            i = self.LOSS_FORMAT % (self.logger.name.upper(), steps, values)
            
        self.logger.info(i)

    def _log_to_same_loger(self, s):
        self.logger.info(s)

def read_step_and_loss(**kwargs):
    result = dict()
    pattern = re.compile(r'\[(.*?)\]')
    for _, name in enumerate(kwargs):
        with open(kwargs[name], mode='r') as f:
            xs = list()
            ys = list()
            for line in f.readlines():
                (x, y) = pattern.findall(line)
                xs.append(float(x))
                ys.append(float(y))
            result[name] = (xs, ys)
    return result

                


if __name__ == "__main__":
    
    '''testing
    '''
    read_step_and_loss(haha='G:\\VS Code\\Unshifting-Mask\\_MLOGS\\18-11-29-20-11\\cross_entropy.log')
    # setup_log_folder_name()
    # log = LLoger("ADAM")
    # log.loss(5, 5.0)


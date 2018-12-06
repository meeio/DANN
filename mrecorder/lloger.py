import logging
import datetime
import os

LOG_DIR = ""
FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

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

def setup_log_folder_name(tag_name = None, base_folder_name="./_MLOGS/", create=True):
    '''Set global path to store log file
    
    Keyword Arguments:
        base_folder_name {string} -- the dir for base folder. (default: {None})
        create {bool} -- create if folder not exist. (default: {True})
    '''

    global LOG_DIR
    LOG_DIR = path_helper(base_folder_name=base_folder_name, tag_name=tag_name)
    

class LLoger(object):
    
    def __init__(self, name , level=logging.INFO, to_file=False):
    
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if to_file:
            if LOG_DIR is "":
                raise Exception(
                    "Please call function <set_new_folder_name> before create a LLoger "
                )
            log_file = LOG_DIR + name + '.log'

            handler = logging.FileHandler(log_file)
            handler.setFormatter(FORMATTER)
            logger.addHandler(handler)


        self.LOSS_FORMAT = "[%s at step %s is %s]"

        self.logger = logger
        self.range_loss = None
        self.range_step = 0

    def record(self, closs):
        if self.range_loss is None:
            self.range_loss = closs
        else:
            self.range_loss += closs
        range_step += 1
    
    def log_current_avg_loss(self, step):
        self.loss(step, self.avg_record())

    def avg_record(self):
        result = self.range_loss / self.range_step
        self.range_loss = None
        self.range_step = 0
        return result

    def loss(self, steps, values):
        '''record loss to log file
        
        Arguments:
            steps {current step} -- steps os current loss
            values {[type]} -- values of current loss
        '''

        i = self.LOSS_FORMAT % (self.logger.name, steps, values)
        self.logger.info(i)



if __name__ == "__main__":
    
    '''testing
    '''

    setup_log_folder_name()
    log = LLoger("ADAM")
    log.loss(5, 5.0)


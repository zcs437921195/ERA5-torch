import datetime
import torch
import os

############################
# Device
############################
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

############################
# train or valid
############################
TRAIN = True


############################
# Data setting
############################
d = [datetime.datetime(2014, 5, 1, 0)]
for i in range(2):
    new_d = d[-1] + datetime.timedelta(days=1)
    d.append(new_d)
DATASET="ERA5"  # ERA5
TRAIN_CFG = dict(
    data_dir="/data-fat01/TianZhiBei/科目二数据/数据/ERA5/", # training dataset path
    level="Ground", # era5 data level. Ground/Pressure.
    dates=[datetime.datetime(2014, 6, 30, 0)] * 8, # use ERA5 dates.
    # dates=d.copy(), # use ERA5 dates.
    elements=["10m_u_component_of_wind"], # use elements. The list of all elements see modules/constants.ERA5_DATA_ELEMENTS
    height=128,
    width=128,
    total_sql_len=24, # length of sequence
    inp_sql_len=23, # length of inputs sequence
    out_sql_len=1, # length of outputs sequence
)
d = [datetime.datetime(2014, 5, 1, 0)]
for i in range(2):
    new_d = d[-1] + datetime.timedelta(days=1)
    d.append(new_d)
VALID_CFG = dict(
    data_dir="/data-fat01/TianZhiBei/科目二数据/数据/ERA5/",
    level="Ground", # era5 data level. Ground/Pressure.
    dates=[datetime.datetime(2014, 6, 30, 0)], # use ERA5 dates.
    # dates=d.copy(), # use ERA5 dates.
    elements=["10m_u_component_of_wind"], # use elements. The list of all elements see modules/constants.ERA5_DATA_ELEMENTS
    height=128,
    width=128,
    total_sql_len=24, # length of sequence
    inp_sql_len=23, # length of inputs sequence
    out_sql_len=1, # length of outputs sequence
)
BATCH_SIZE = 4


############################
# Training setting
############################
BLR = 1e-2
LR_RATE = BLR * BATCH_SIZE / 256
EPOCH = 100
VALID_EPOCH = 1 # vallidation every VALID_EPOCH


############################
# Log setting
############################
now = datetime.datetime.now()
now = now.strftime("%Y%m%d%H%M%S")
OUTPUTS_PATH = os.path.join('./outputs/', now)
SUMMARY_PATH = os.path.join(OUTPUTS_PATH, "summary")
LOG_FILE = os.path.join(OUTPUTS_PATH, 'log.out')
LOG_COL = ('Epochs', 'Training Loss', 'Evaluation', 'Time')
LOG_COL_LEN = (10, 36, 36, 36)
USE_TENSORBOARD = True


############################
# Visualize setting
############################
VISUALIZE = True
VISUAL_PATH = os.path.join(OUTPUTS_PATH, 'visual')



############################
# Update configs from parser
############################
def update_config(args, config):
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        config.LR_RATE = config.BLR * config.BATCH_SIZE / 256
    if args.train_data_dir is not None:
        config.TRAIN_CFG["data_dir"] = args.train_data_dir
    if args.valid_data_dir is not None:
        config.VALID_CFG["data_dir"] = args.valid_data_dir
    if args.log is not None:
        config.OUTPUTS_PATH = os.path.join(os.path.dirname(config.OUTPUTS_PATH), args.log)
        config.SUMMARY_PATH = os.path.join(config.OUTPUTS_PATH, "summary")
        config.LOG_FILE = os.path.join(config.OUTPUTS_PATH, 'log.out')
    return config
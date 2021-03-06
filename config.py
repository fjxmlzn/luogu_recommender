# database
HOST = "localhost"
USER = "root"
PASSWORD = "root"
DB = "luogu"
PORT = 3306
CHARSET = "utf8"

# system parameters
REC_MATRIX_FILE = "rec_matrix.pkl"

# train parameters
TRAIN_WEIGHT = [0, 0.8, 0.4, 0.2, 0.1]

# test parameters
DECAY_TIME = 7 * 24 * 60 * 60
DECAY_RATE = 0.8
DIFFICULTY_ARGUMENT_FACTOR = 1.2
DIFFICULTY_EXP_ALPHA = 0.5
DIFFICULTY_EXP_MIN = 0.2

# validate parameters
HISTORY_LENGTH = 3


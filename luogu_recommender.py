import pymysql
import pymysql.cursors
import sys, os, pickle, math, time
import numpy as np
import matplotlib.pyplot as plt
from config import *


pub_upids_dict = {}
pub_upids_list = []
pub_difficulty_list = []
rec_matrix = None

def print_error(str):
    sys.stderr.write(str)


def print_log(str):
    sys.stdout.write(str)


"""
Train recommendation parameters, which is saved in REC_MATRIX_FILE.
- Input: None
- Output: None
"""
def train():
    connection=pymysql.connect(host = HOST,
                               user = USER,
                               password = PASSWORD,
                               db = DB,
                               port = PORT,
                               charset = CHARSET)
    try:
        # get all public problems
        with connection.cursor() as cursor:
            sql = "select upid, difficulty from problem where type=1"
            cursor.execute(sql)
            pub_upids = cursor.fetchall()
        pLen = len(pub_upids)
        pub_upids_dict = {}
        pub_upids_list = []
        pub_difficulty_list = []
        for i in range(pLen):
            pub_upids_list.append(pub_upids[i][0])
            pub_upids_dict[pub_upids[i][0]] = i
            pub_difficulty_list.append(pub_upids[i][1])
        assert(pLen == len(pub_upids_dict))
        assert(pLen == len(pub_upids_list))
        print_log("Found {} public problems\n".format(pLen))
        
        # get all users
        with connection.cursor() as cursor:
            sql = "select uid from user"
            cursor.execute(sql)
            uids = cursor.fetchall()
        uLen = len(uids)
        print_log("Found {} users\n".format(uLen))
        
        # train the recommendation parameters by records of each user
        rec_matrix = np.zeros((pLen, pLen))
        num_pairs = 0
        for i in range(uLen):
            print_log("Processing {}/{}...\r".format(i, uLen))
            with connection.cursor() as cursor:
                sql = "select upid from record where `uid` = %s order by rid asc"
                cursor.execute(sql, uids[i])
                upids = cursor.fetchall()
            filtered_upids_list = []
            for upid in upids:
                if upid[0] in pub_upids_dict \
                and (not filtered_upids_list or filtered_upids_list[-1] != upid[0]):
                    filtered_upids_list.append(upid[0])
            for i in range(1, len(filtered_upids_list)):
                for j in range(max(0, i - len(TRAIN_WEIGHT) + 1), i):
                    if (filtered_upids_list[i] != filtered_upids_list[j]):
                        rec_matrix[pub_upids_dict[filtered_upids_list[j]]][pub_upids_dict[filtered_upids_list[i]]] += TRAIN_WEIGHT[i - j]
                        num_pairs += 1
        print_log("Found {} valid record pairs\n".format(num_pairs))
        
        for i in range(pLen):
            max_value = np.max(rec_matrix[i,:])
            if max_value > 0:
                rec_matrix[i,:] = rec_matrix[i,:] / max_value
        
        f = open(REC_MATRIX_FILE, "wb")
        pickle.dump(rec_matrix, f)
        pickle.dump(pub_upids_list, f)
        pickle.dump(pub_upids_dict, f)
        pickle.dump(pub_difficulty_list, f)
        f.close()
        print_log("Finished training\n")

    finally:
        connection.close()
        

"""
Recommend the suitable problems for the user to do.
- Input: 
    * the list of recent problems that the user did in format of (upid, timestamp)
      from the latest to the oldest
      <those problems will not exist in the output list>
    * number of returned problems. -1 suggests all
    * other problems that should not exist in the output list
- Output: 
    * the list of suggested problems, starting from the best
"""
def recommend(history, num = -1, remove = []):
    global rec_matrix, pub_upids_list, pub_upids_dict, pub_difficulty_list
    current_time = time.time()
    
    def get_weight(time_difference):
        return DECAY_RATE ** (time_difference / DECAY_TIME)
    
    if not pub_upids_list:
        if not os.path.isfile(REC_MATRIX_FILE):
            print_error("Please run function `train` first")
            return
        f = open(REC_MATRIX_FILE, "rb")
        rec_matrix = pickle.load(f)
        pub_upids_list = pickle.load(f)
        pub_upids_dict = pickle.load(f)
        pub_difficulty_list = pickle.load(f)
        f.close()
        
    remove_dict = {}
    for p in remove:
        remove_dict[p] = True
        
    filtered_history = []
    for (p, t) in history:
        if p in pub_upids_dict \
        and (not filtered_history or filtered_history[-1][0] != p):
            filtered_history.append((p, t))
            remove_dict[p] = True
    
    difficulty_sum = 0
    difficulty_num = 0
    for (p, t) in filtered_history:
        if pub_difficulty_list[pub_upids_dict[p]] is not None\
        and pub_difficulty_list[pub_upids_dict[p]] > 0:
            weight = get_weight(current_time - t)
            difficulty_sum += pub_difficulty_list[pub_upids_dict[p]] * weight
            difficulty_num += weight
    if difficulty_num > 0:
        difficulty_mean = DIFFICULTY_ARGUMENT_FACTOR * difficulty_sum / difficulty_num
    else:
        difficulty_mean = -1
    
    score = np.zeros((len(pub_upids_list), ))
    for (p, t) in filtered_history:
        score += rec_matrix[pub_upids_dict[p]][:] * get_weight(current_time - t)
    
    if difficulty_mean != -1:
        for i in range(len(pub_upids_list)):
            if pub_difficulty_list[i] is not None and pub_difficulty_list[i] > 0:
                score[i] *= DIFFICULTY_EXP_MIN + math.exp(-DIFFICULTY_EXP_ALPHA * abs(difficulty_mean - pub_difficulty_list[i])) * (1 - DIFFICULTY_EXP_MIN)
    
    recommendation = []
    ids = np.argsort(-score)
    for id in ids:
        if (score[id] == 0):
            break
        if pub_upids_list[id] in remove_dict:
            continue
        recommendation.append(pub_upids_list[id])
    
    if num == -1:
        return recommendation
    else:
        return recommendation[0 : num]
        

"""
Validate the recommendation results on all records.
- Input: None
- Output: None
"""
def validate():
    connection=pymysql.connect(host = HOST,
                               user = USER,
                               password = PASSWORD,
                               db = DB,
                               port = PORT,
                               charset = CHARSET)
    try:
        # get all users
        with connection.cursor() as cursor:
            sql = "select uid from user"
            cursor.execute(sql)
            uids = cursor.fetchall()
            #uids = cursor.fetchmany(10)
        uLen = len(uids)
        print_log("Found {} users\n".format(uLen))
        
        # analysis recommendation results for each user
        all_scores = []
        for i in range(uLen):
            single_scores = []
            
            with connection.cursor() as cursor:
                sql = "select upid, submittime from record where `uid` = %s order by rid asc"
                cursor.execute(sql, uids[i])
                upids = cursor.fetchall()
                
            filtered_upids_list = []
            for upid in upids:
                if not filtered_upids_list or filtered_upids_list[-1] != upid[0]:
                    filtered_upids_list.append((upid[0], upid[1]))
                
            for j in range(HISTORY_LENGTH, len(filtered_upids_list) - 1):
                t = filtered_upids_list[j - HISTORY_LENGTH : j]
                t.reverse()
                result = recommend(t)
                if filtered_upids_list[j][0] in result:
                    score = result.index(filtered_upids_list[j][0]) + 1
                    single_scores.append(score)
            print_log("Processing {} / {}: {} ({} records)\n".format(i, uLen, "N/A" if len(single_scores) == 0 else np.mean(single_scores), len(single_scores)))
            
            all_scores.extend(single_scores)
        
        print_log("Finished validation: {} ({} records)\n".format("N/A" if len(all_scores) == 0 else np.mean(all_scores), len(all_scores)))
        
        f = open("temp.pkl", "wb")
        pickle.dump(all_scores, f)
        f.close()
        
        plt.hist(all_scores, 800)
        plt.show()

    finally:
        connection.close()
        

if __name__ == "__main__":
    train()
    validate()
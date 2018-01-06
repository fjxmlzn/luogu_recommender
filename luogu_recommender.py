import pymysql
import pymysql.cursors
import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from config import *


pub_upids_dict = {}
pub_upids_list = []
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
            sql = "select upid from problem where type=1"
            cursor.execute(sql)
            pub_upids = cursor.fetchall()
        pLen = len(pub_upids)
        pub_upids_dict = {}
        pub_upids_list = []
        for i in range(pLen):
            pub_upids_list.append(pub_upids[i][0])
            pub_upids_dict[pub_upids[i][0]] = i
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
        
        f = open(REC_MATRIX_FILE, "wb")
        pickle.dump(rec_matrix, f)
        pickle.dump(pub_upids_list, f)
        pickle.dump(pub_upids_dict, f)
        f.close()
        print_log("Finished training")

    finally:
        connection.close()
        

"""
Recommend the suitable problems for the user to do.
- Input: 
    * the list of recent problems that the user did, from the latest to the oldest
    * number of returned problems. -1 suggests all
- Output: 
    * the list of suggested problems, starting from the best
"""
def recommend(history, num = -1):
    global rec_matrix, pub_upids_list, pub_upids_dict
    if not pub_upids_list:
        if not os.path.isfile(REC_MATRIX_FILE):
            print_error("Please run function `train` first")
            return
        f = open(REC_MATRIX_FILE, "rb")
        rec_matrix = pickle.load(f)
        pub_upids_list = pickle.load(f)
        pub_upids_dict = pickle.load(f)
        f.close()
        
    filtered_history = []
    for p in history:
        if p in pub_upids_dict \
        and (not filtered_history or filtered_history[-1] != p):
            filtered_history.append(p)
    
    score = np.zeros((len(pub_upids_list), ))
    for i in range(len(filtered_history)):
        score += rec_matrix[pub_upids_dict[filtered_history[i]]][:] * (DECAY_RATE ** i)
    
    recommendation = []
    ids = np.argsort(-score)
    for id in ids:
        if (score[id] == 0):
            break
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
                sql = "select upid from record where `uid` = %s order by rid asc"
                cursor.execute(sql, uids[i])
                upids = cursor.fetchall()
                
            filtered_upids_list = []
            for upid in upids:
                if not filtered_upids_list or filtered_upids_list[-1] != upid[0]:
                    filtered_upids_list.append(upid[0])
                
            for j in range(HISTORY_LENGTH, len(filtered_upids_list) - 1):
                t = filtered_upids_list[j - HISTORY_LENGTH : j]
                t.reverse()
                result = recommend(t)
                if filtered_upids_list[j] in result:
                    score = result.index(filtered_upids_list[j]) + 1
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
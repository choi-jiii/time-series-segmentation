"""
TICC를 읽어오지 못하는 문제(경로문제)
=> ticc-main의 상위 폴더인 TICC부터 불러오는 경우 발생(같은 이름이 원인인 것으로 보임)
=> TICC가 아닌 ticc-main 폴더를 open
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from ticc import TICC
import datetime
import time


# 최대 출력 제한 해제(Option)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)

start = time.time()

try:
    path = Path(__file__).absolute().parent.parent
except NameError:
    path = Path().absolute()

# User Parameters--------------------------------------------------------
feature_cnt = 100
window_size = 5
n_clusters = 6
# -----------------------------------------------------------------------

result_path =  'results/feature{}/'.format(feature_cnt)

fname = path / 'data/HAR/data_asc.csv'

# 출력 파일 생성
iter_log = result_path + 'iter_log_f{}_w{}_c{}.txt'.format(feature_cnt, window_size, n_clusters)
sys.stdout = open(iter_log, 'w')

X = pd.read_csv(fname).iloc[:, :feature_cnt].values
np.random.seed(102)

model = TICC(
    window_size=window_size, # sliding window 크기
    n_clusters=n_clusters, # cluster 개수
    lambda_parameter=11e-2, # 정규화 매개변수(각 클러스터의 sparsity를 결정함) 11e-2 : 0.11
    switch_penalty=600,
    max_iters=100, # 최대 반복 횟수(default: 100) -> primal and dual residual value가 0에 수렴하면 stop
    threshold=2e-5, # 수렴 임계값 2e-5 :  2 * 10^-5 = 2 * 0.00001 = 0.00002
    n_jobs=-1, # 생성할 프로세스 수
    verbose=1 # int, 출력 상세 수준(1: 중간 수준)
)

model.fit(X)

elapsed_time_seconds = time.time() - start
elapsed_time = str(datetime.timedelta(seconds=elapsed_time_seconds))
print("time : ", elapsed_time)

# 출력 파일 저장
sys.stdout.close()

cluster_indices = model.get_cluster_indices()
df = pd.DataFrame(cluster_indices.items(), columns=['Cluster', 'Indices'])

# CSV 파일로 저장
df.to_csv(result_path + 'cluster_results_f{}_w{}_c{}.csv'.format(feature_cnt, window_size, n_clusters), index=False)

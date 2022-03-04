from math import fabs
from tqdm import tqdm
from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import csv
import pickle
f = open("love.txt",'a')
writer = csv.writer(f)
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
def store_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
l=[]
for i in tqdm(range(10000000)):
    writer.writerow([i,i*2])
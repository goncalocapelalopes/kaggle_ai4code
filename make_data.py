import json
import os

import pandas as pd
from torch.utils.data import Dataset

GEN_INTERIM = False
TRAIN_PATH = os.path.join("data", "raw", "train")
TEST_PATH = os.path.join("data", "raw", "test")

INTERIM_PATH = os.path.join("data", "interim")

TRAIN_ORDERS = os.path.join("data", "raw", "train_orders.csv")

class NotebookDataset(Dataset):
    def __init__(self, orders_file, notebook_dir):
        self.notebook_orders = pd.read_csv(orders_file)
        self.notebook_dir = notebook_dir
    
    def __len__(self):
        return len(self.notebook_orders)

def json2df(jsonpath, order=None):
    with open(jsonpath, "r") as f:
        dct = json.loads(f.read())
    
    ids = []
    text = []
    if order is not None:
        for e in order:
            ids.append(e)
            text.append(dct["source"][e])
    else:
        ids = list(dct["source"].keys())
        text = list(dct["source"].values())
    return pd.DataFrame(data={"id": ids, "text": text})
        
def gen_data(train_dir=TRAIN_PATH, orders_file=TRAIN_ORDERS):
    orders = pd.read_csv(orders_file)
    for idx, row in orders.iterrows():
        jsonpath = os.path.join(TRAIN_PATH, 
                                orders.iloc[idx, 0]+".json")
        correct_order = json2df(jsonpath, row["cell_order"].split(" "))
        wrong_order = json2df(jsonpath)
        
        correct_order["label"] = 1
        wrong_order["label"] = 0

        outpath = os.path.join(INTERIM_PATH, orders.iloc[idx, 0] + "{}.csv")
        for j, o in zip(["_correct", "_wrong"], [correct_order, wrong_order]):
            o.to_csv(outpath.format(j))


if __name__ == "__main__":
    if GEN_INTERIM:
        gen_data()
        
import pandas as pd 
import os 
import glob 
import argparse
from pathlib import Path

dir_path = Path().resolve()
project_root = dir_path.parent.parent


def combine_data():
    all_files = glob.glob(os.path.join(project_root / "Dataset" / "raw" / "Wind Farm A" / "datasets" / "*.csv"
))
    dfs = []
    for filename in all_files: 
       temp = pd.read_csv(filename, index_col=None, header=0)
       dfs.append(temp)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(project_root + "/Dataset/combined/df_all.csv", index=False, sep=",")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=False)
    ap.add_argument("--out_dir", type=str, required=False)

    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.outpath)
    out_dir.mkdir(parents=True, exist_ok=True)
    combine_data()

if __name__ == "__main__":
    main() 
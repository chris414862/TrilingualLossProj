import argparse
import re
from pathlib import Path
import pandas as pd
import sys


parser = argparse.ArgumentParser(description='Produce Results in table form')
parser.add_argument('dir_list', help='Path to file containing newline separated experiment directory names for display in table')
parser.add_argument('dir_path', help='Path to file containing newline separated experiment directory names for display in table')
parser.add_argument('savename', help='Filename to save output.')
parser.add_argument('--omit_cos_sim', action="store_true",
                help='Do not include cosine similarity results')
args = parser.parse_args()

def filter_cols(regex, cols):
    return [s for s in filter(lambda x: re.search(regex, x), cols)]

def colate_cols(regex_list, sorted_cols):
    filtered_col_sets = []
    for regex in regex_list:
        filtered_col_sets.append(filter_cols(regex, sorted_cols))

    colated_cols = []
    [colated_cols.extend(cols) for cols in zip(*filtered_col_sets)]
    return colated_cols

def get_col_sequence(sorted_cols, sim_type='dot'):
    """
        Organizes the sequence of columns in the table
    """
    regex_list = [r".*image.*"+sim_type+"_avg_r1$", r".*image.*"+sim_type+"_avg_r5$", r".*image.*"+sim_type+"_avg_r10$"]
    img_avg_cols = colate_cols(regex_list, sorted_cols)

    regex_list = [r".*(hindi|japanese)&english.*"+sim_type+"_avg_r1$", r".*(hindi|japanese)&english.*"+sim_type+"_avg_r5$", r".*(hindi|japanese)&english.*"+sim_type+"_avg_r10$"]
    english_avg_cols = colate_cols(regex_list, sorted_cols)

    regex_list = [r".*(english|japanese)&hindi.*"+sim_type+"_avg_r1$", r".*(english|japanese)&hindi.*"+sim_type+"_avg_r5$", r".*(english|japanese)&hindi.*"+sim_type+"_avg_r10$"]
    hindi_avg_cols = colate_cols(regex_list, sorted_cols)

    regex_list = [r".*(hindi|english)&japanese.*"+sim_type+"_avg_r1$", r".*(hindi|english)&japanese.*"+sim_type+"_avg_r5$", r".*(hindi|english)&japanese.*"+sim_type+"_avg_r10$"]
    j_avg_cols = colate_cols(regex_list, sorted_cols)
    
    return img_avg_cols + english_avg_cols + hindi_avg_cols + j_avg_cols


with open(args.dir_list, "r") as f:
    dir_path = Path(args.dir_path)
    exp_dirs = [dir_path / line.strip() for line in f.readlines()]


exp_df_dict = {}
for exp_dir in exp_dirs:
    progress_file = exp_dir / "progress.pkl"
    df = pd.read_pickle(progress_file)
    if (exp_dir / "description.txt").exists():
        with open(exp_dir / "description.txt", "r") as f:
            descrip = " ".join([line.rstrip() for line in f.readlines()])
    else:
        descrip = ""
    exp_df_dict[exp_dir] ={"df": df, "descrip":descrip}

exp_best_ep_dict = {}
use_cos_sim = True
row_data = {}
pd.set_option("display.max_rows", 500)
for exp_dir, d in exp_df_dict.items():
    df, descrip = d['df'], d['descrip']
    best_epoch = int(df.iloc[-1]["best_epoch"])-1
    sorted_cols = [item[0] for item in sorted(df.iloc[best_epoch].iteritems(), key=lambda x: x[0]) if re.match(r"^.*(_avg_|_dot_|_cos_)", item[0]) is not None]
    sorted_cols = [c for c in sorted_cols if not re.search(r"_norm_dot_", c)]
     
    col_sequence = get_col_sequence(sorted_cols, sim_type="dot")
    if not args.omit_cos_sim:
        col_sequence +=  get_col_sequence(sorted_cols, sim_type="cos")


    best_ep_results = df.iloc[best_epoch]
    # Additional info
    add_info = pd.Series({
                "Description":descrip,
                "Best Epoch":best_epoch
                })
    add_info_keys = add_info.index.tolist()
    best_ep_results = best_ep_results.append(add_info)

    col_sequence = add_info_keys+col_sequence
    curated_ep_results = best_ep_results[col_sequence]


    reduced_col_sequence = []
    for col in col_sequence:
        col = re.sub(r"image","img", col)
        col = re.sub(r"japanese","jap", col)
        col = re.sub(r"hindi","hin", col)
        col = re.sub(r"english","eng", col)
        reduced_col_sequence.append(col)

    curated_ep_results.index = reduced_col_sequence
    # Key is experiment directory name, value is row of results
    row_data[exp_dir.parts[-1]] = curated_ep_results


# Orientating by index to make formating easier by column 
# Pandas has special enforcement rules for data types in columns, so changing dt by rows is difficult
df = pd.DataFrame.from_dict(row_data, orient="index")
for col in df.columns:
    if col in add_info_keys:
        continue
    df[col] = pd.Series([f"{val:.2%}" for k, val in df[col].iteritems()], index=df.index)

df.T.to_csv(args.savename)
print(df.T)
    
    

        







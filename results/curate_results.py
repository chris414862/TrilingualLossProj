import argparse
import re
from pathlib import Path
import pandas as pd
import sys
from itertools import zip_longest


parser = argparse.ArgumentParser(description='Produce Results in table form')
parser.add_argument('dir_list', help='Path to file containing newline separated experiment directory names for display in table')
parser.add_argument('dir_path', help='Path to file containing newline separated experiment directory names for display in table')
parser.add_argument('--savename', help='Filename to save output.')
# parser.add_argument('--omit-cos-sim', action="store_true",
#                 help='Do not include cosine similarity results')
parser.add_argument('--group-by-task', action="store_true",
                help='Group results by view pair')
parser.add_argument('--include-all', action="store_true",
                help='Include directions with average')
parser.add_argument('--only-directions', action="store_true",
                help='Only include directions')
args = parser.parse_args()

# def filter_cols(regex, cols):
#     return [s for s in filter(lambda x: re.search(regex, x), cols)]

def colate_cols(regex_list, sorted_cols):
    filtered_col_sets = []
    for regex in regex_list:
        print(regex)
        filtered_col_sets.append(list(filter(lambda x: re.search(regex, x), sorted_cols)))

    # print(filtered_col_sets)
    colated_cols = []
    [colated_cols.extend(cols) for cols in zip(*filtered_col_sets)]
    return colated_cols




def get_col_sequence(sorted_cols, sim_type='dot', avg_only=False, group_by_task=True):
    """
        Organizes the sequence of columns in the table
    """
    #### Shows only averages
    if avg_only:
        col_regexes = [r".*image.*", r".*(hindi|japanese)&english.*", r".*(english|japanese)&hindi.*", r".*(hindi|english)&japanese.*"]
        nested_regex_list = []
        for col_regex in col_regexes:

            nested_regex_list.append([col_regex+sim_type+"_avg_r" + re.escape(str(i)) +"$" for i in [1,5,10]])
        if group_by_task:
            cols = [col  for regex_list in nested_regex_list for regex in regex_list for col in list(filter(lambda x: re.search(regex, x), sorted_cols))]

        else:
            regex_by_recall = [regex for tup in zip(*nested_regex_list) for regex in tup]
            cols = [col  for regex in regex_by_recall for col in list(filter(lambda x: re.search(regex, x), sorted_cols))]

    #### Shows both directions and averages
    else:
        nested_col_regexes = [[r".*english&image.*", r".*image->english.*", r".*english->image.*"],
                              [r".*hindi&image.*", r".*image->hindi.*", r".*hindi->image.*"],
                              [r".*japanese&image.*", r".*image->japanese.*", r".*japanese->image.*"],
                              [r".*(english&hindi|hindi&english).*", r".*hindi->english.*", r".*english->hindi.*"],
                              [r".*(english&japanese|japanese&english).*", r".*japanese->english.*", r".*english->japanese.*"],
                              [r".*(hindi&japanese|japanese&hindi).*", r".*japanese->hindi.*", r".*hindi->japanese.*"],
                             ]
        triple_nested_regex_list = []
        for col_regexes in nested_col_regexes:
            nested_regex_list = []
            for col_regex in col_regexes:
                nested_regex_list.append([col_regex+sim_type+".*r" + re.escape(str(i)) +"$" for i in [1,5,10]])
            triple_nested_regex_list.append(nested_regex_list)


        cols = []
        if group_by_task:
            for nested_regex_list in triple_nested_regex_list:
                for regex_list in nested_regex_list:
                    cols.extend( [col  for regex in regex_list for col in list(filter(lambda x: re.search(regex, x), sorted_cols))])
        else:
            triple_nested_cols = []
            for nested_regex_list in triple_nested_regex_list:
                nested_cols = []
                for regex_list in zip(*nested_regex_list):
                    nested_cols.append( [col  for regex in regex_list for col in list(filter(lambda x: re.search(regex, x), sorted_cols))])
                triple_nested_cols.append(nested_cols)

                
            for nested_cols in zip(*triple_nested_cols):
                for cur_cols in nested_cols:
                    cols.extend(cur_cols)




             

        #     print(sorted_cols)
        #     print(triple_nested_regex_list)
        #     regex_by_recall = [print(tup) for nested_regex_list in triple_nested_regex_list for tup in zip(*nested_regex_list)]
        #     print(regex_by_recall)
        #     # cols = []
        #     # for regex in regex_by_recall: 
        #     #     print("\n", regex)
        #     #     print(list(filter(lambda x: re.search(regex, x), sorted_cols)))
        #     #     cols.append(list(filter(lambda x: re.search(regex, x), sorted_cols)))
        #     # cols = list(filter(lambda x: not x is None, [tup for tup in zip_longest(*cols)][0]))
        #     # print(cols)
        #     sys.exit()
        # regex_list = [r".*image.*"+sim_type+".*r1$", r".*image.*"+sim_type+".*r5$", r".*image.*"+sim_type+".*r10$"]
        # img_avg_cols = colate_cols(regex_list, sorted_cols)
        #
        # regex_list = [r".*(hindi|japanese)&english.*"+sim_type+".*r1$", r".*(hindi|japanese)&english.*"+sim_type+".*r5$", r".*(hindi|japanese)&english.*"+sim_type+".*r10$"]
        # english_avg_cols = colate_cols(regex_list, sorted_cols)
        #
        # regex_list = [r".*(english|japanese)&hindi.*"+sim_type+".*r1$", r".*(english|japanese)&hindi.*"+sim_type+".*r5$", r".*(english|japanese)&hindi.*"+sim_type+".*r10$"]
        # hindi_avg_cols = colate_cols(regex_list, sorted_cols)
        #
        # regex_list = [r".*(hindi|english)&japanese.*"+sim_type+".*r1$", r".*(hindi|english)&japanese.*"+sim_type+".*r5$", r".*(hindi|english)&japanese.*"+sim_type+".*r10$"]
        # j_avg_cols = colate_cols(regex_list, sorted_cols)
    
    return cols


with open(args.dir_list, "r") as f:
    dir_path = Path(args.dir_path)
    exp_dirs = []
    sim_types = []
    pref_names = []
    for line in f.readlines():
        toks = line.split()
        exp_dir = dir_path / toks[0].strip()
        exp_dirs.append(exp_dir)
        if len(toks) > 1:
            for tok in toks[1:]:
                if tok.lower() in ["dot", "cos"]: # similarity measure to use
                    sim_types.append(tok)
                else: # preferred name of experiment
                    pref_names.append(re.sub(r"_", " ",tok))
        if len(sim_types) != len(exp_dirs):
            sim_types.append("dot")

        if len(pref_names) != len(exp_dirs):
            pref_names.append(exp_dir)
    # exp_dirs = [dir_path / line.strip() for line in f.readlines()]


exp_df_dict = {}
for exp_dir,sim_type, pref_name in zip(exp_dirs,sim_types, pref_names):
    if exp_dir == "":
        continue


    print(f"Searching directory: {exp_dir}")
    # Get progress dataframe
    progress_file = exp_dir / "progress.pkl"
    df = pd.read_pickle(progress_file)

    # Get description (if any)
    if (exp_dir / "description.txt").exists():
        with open(exp_dir / "description.txt", "r") as f:
            descrip = " ".join([line.rstrip() for line in f.readlines()])
    else:
        descrip = ""

    exp_df_dict[exp_dir] ={"df": df, "descrip":descrip, "sim_type": sim_type, "pref_name": pref_name}

exp_best_ep_dict = {}
use_cos_sim = True
row_data = {}
pd.set_option("display.max_rows", 500)
for exp_dir, d in exp_df_dict.items():
    df, descrip = d['df'], d['descrip']
    best_epoch = int(df.iloc[-1]["best_epoch"])-1
    sorted_cols = [item[0] for item in sorted(df.iloc[best_epoch].iteritems(), key=lambda x: x[0]) if re.match(r"^.*(_avg_|_dot_|_cos_)", item[0]) is not None]
    sorted_cols = [c for c in sorted_cols if not re.search(r"_norm_dot_", c)]
     
    col_sequence = get_col_sequence(sorted_cols, sim_type=d['sim_type'], group_by_task=args.group_by_task, avg_only=not args.include_all and not args.only_directions)
    # if not args.omit_cos_sim:
    #     col_sequence +=  get_col_sequence(sorted_cols, sim_type="cos", group_by_task=args.group_by_task)


    best_ep_results = df.iloc[best_epoch]
    # Additional info
    add_info = pd.Series({
                "Description":descrip,
                "Ep":best_epoch
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
        col = re.sub(r"_cos", "", col)
        col = re.sub(r"_dot", "", col)
        reduced_col_sequence.append(col)



    curated_ep_results.index = reduced_col_sequence
    # Key is experiment directory name, value is row of results
    row_data[d["pref_name"]] = curated_ep_results


# Orientating by index to make formating easier by column 
# Pandas has special enforcement rules for data types in columns, so changing dt by rows is difficult
df = pd.DataFrame.from_dict(row_data, orient="index")



# Format column names:
new_cols = []
for col_name in df.columns:
    new_name = re.sub(r"hin", "H", col_name) 
    new_name = re.sub(r"jap", "J", new_name)
    new_name = re.sub(r"eng", "E", new_name)
    new_name = re.sub(r"img", "I", new_name)
    new_name = re.sub(r"_(dot|cos)_", ".", new_name)
    new_name = re.sub(r"_(dot|cos)_", ".", new_name)
    new_name = re.sub(r"_r", "R", new_name)
    new_name = re.sub(r"\.r", ".R", new_name)
    new_name = re.sub(r"_avg", ".", new_name)

    new_cols.append(new_name)

df.columns = new_cols
# sort cols by image retrieval vs cross lingual
image_ret_cols = list(filter(lambda x: re.search(r"(&I|->I|I->)", x), df.columns))
cross_ling_cols = list(filter(lambda x: not x in image_ret_cols+["Ep", "Description"], df.columns))

if args.only_directions:
    image_ret_cols = list(filter(lambda x: re.search(r"(->|<-)", x), image_ret_cols))
    cross_ling_cols = list(filter(lambda x: re.search(r"(->|<-)", x), cross_ling_cols)) 
df1 = df[image_ret_cols + ["Ep"]]
df2 = df[cross_ling_cols + ["Ep"]]



if not args.savename is None:
    df.to_csv(args.savename)

# Formatting results to be in percents
def format_percents(df, maxes=None):
    for col in df.columns:
        if col in add_info_keys:
            continue
        df[col] = pd.Series([f"{val:.2%}" if idx != maxes[col] else "textbf"+f"{{{val:.2%}}}" for idx, val in df[col].iteritems()], index=df.index)

    return df

def final_formatting(latex_str, res_type="Image Retrieval", remove_top=False, remove_bottom=False):

    formatted_latex_str = ""
    for string in latex_str.split("\n"):
        if remove_top and re.search(r"begin\{tabular", string):
            continue
        if remove_bottom and re.search(r"end\{tabular", string):
            continue
        if re.search(r"Ep", string):
            num_cols = len(re.split(r" & ", string))-1
            formatted_latex_str += re.sub(r"\{\}",f"{res_type}", string) +"\n" 
            #+"\n"+res_type+"&"*num_cols+"\\\\\n\midrule\n"
        # elif re.search(r"Description", string):
        #     formatted_latex_str += "\midrule\n Cross-Ling\n\midrule\n"

        else:
            formatted_latex_str = re.sub(r"textbf", re.escape("\\bf"), formatted_latex_str)
            formatted_latex_str = re.sub(r"\\(\{|\})", r"\1", formatted_latex_str)
            formatted_latex_str +=string+"\n"
    return formatted_latex_str.strip()

print("INFO")
print(df1.dtypes)
print("idxmax")
print(df1.idxmax())
df1_maxes = df1.idxmax()
df2_maxes = df2.idxmax()
df1 = format_percents(df1, maxes=df1_maxes)
df2 = format_percents(df2, maxes=df2_maxes)
print(df1)
print(df2)
print()
df1_latex_str = (df1).to_latex()
df2_latex_str = (df2).to_latex()
df1_formatted_str = final_formatting(df1_latex_str, res_type="Image Ret", remove_bottom=True)
df2_formatted_str = final_formatting(df2_latex_str, res_type="Cross-Ling", remove_top=True)
print(df1_formatted_str)
print(df2_formatted_str)

    

        







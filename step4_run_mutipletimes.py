import os
from tqdm import tqdm
import glob
import os
import json


names = [
    "qiuxiao",
    "neixinlun",
    "hudiejian",
    "daoliuzhao",
    "banjinjia",
    "diaohuanluoshuan",
    "yuanguan",
    "lianjiejian",
    "hudiebanjin",
    "banjinjianlong",
    "zhijiaobanjin",
    "jingjiagongjian",
    "jiaojieyuanguan",
    "ganqiuxiao",
    "fanguangzhao",
]

for dataset_name in tqdm(names):
    # for sam + pbr
    command = f"python run_inference.py dataset_name={dataset_name}"
    print(command)  # Print the command to verify
    os.system(command)  # Executes the command in the shell

    # # Fast sam + pbr
    # command = f"python run_inference.py dataset_name={dataset_name} model=cnos_fast"
    # print(command)  # Print the command to verify
    # os.system(command)  # Executes the command in the shell

    # # For pyrender
    # command = f"python run_inference.py dataset_name={dataset_name} model.onboarding_config.rendering_type=pyrender" 
    # print(command)  # Print the command to verify
    # os.system(command)  # Executes the command in the shell

# Step 5 saved into 1 file
saved_jsons = glob.glob("datasets/bop23_challenge/results/cnos_exps/*.json")
# saved_jsons = glob.glob("/home/cuong.van-dam/CuongVanDam/do_an_tot_nghiep/cnos/datasets/bop23_challenge/results/cnos_bow_final_test_top_1/*.json")
print("saving: ", len(saved_jsons))

results = list()
for saved_json in saved_jsons:
   result = json.load(open(saved_json, 'r'))
   results = results + result

output_path = "evaluations/cnos_final_xyz_testset.json"
# output_path = "evaluations/cnos_bow_final_test_top_1.json" # the last fangua left
with open(output_path, 'w') as f:
    json.dump(results, f) # indent=4)


# then tried for cnos_bows_new_method with differnt aggreation

# cnos_bows_new_method_2 same as sam6d as chooing the best template from CNOS and then calcualte the score of BoWs of just this tempalte and average with CNOS score

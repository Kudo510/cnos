import os
import subprocess

names = [
    # "qiuxiao", "neixinlun",
    # "hudiejian", "daoliuzhao", "banjinjia",
    # "diaohuanluoshuan", "yuanguan", "lianjiejian", "hudiebanjin", 
    # "banjinjianlong", "zhijiaobanjin", "jingjiagongjian", 
    "jiaojieyuanguan", 
    # "ganqiuxiao", 
    # "fanguangzhao"
]

models = ["sam6d"] # ["cnos", "sam6d" , "cnos_bows"]

def get_input_file(model, dataset_name):
    base_path = "/home/cuong.van-dam/CuongVanDam/do_an_tot_nghiep"
    
    if model == "cnos":
        return f"{base_path}/cnos/datasets/bop23_challenge/results/cnos_final_test/CustomSamAutomaticMaskGenerator_template_pbr1_aggavg_5_{dataset_name}.json"
    
    elif model == "sam6d":
        return f"{base_path}/Sam6D/SAM-6D/Instance_Segmentation_Model/log/sam_final_test/result_{dataset_name}.json"
    
    elif model == "cnos_bows":
        return f"{base_path}/cnos/datasets/bop23_challenge/results/cnos_bow_final_test/CustomSamAutomaticMaskGenerator_template_pbr1_aggavg_5_{dataset_name}.json"

def main():
    for model in models:
        for dataset_name in names:
            input_file = get_input_file(model, dataset_name)
            output_dir = f"/home/cuong.van-dam/CuongVanDam/do_an_tot_nghiep/cnos/visualize_all/{dataset_name}_{model}_threshold_04"
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Run visualization command
            command = [
                "python", "-m", "src.scripts.visualize", 
                f"dataset_name={dataset_name}", 
                f"input_file={input_file}", 
                f"output_dir={output_dir}"
            ]
            
            subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
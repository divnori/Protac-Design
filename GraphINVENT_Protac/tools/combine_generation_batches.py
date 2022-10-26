"""
Combines .likelihood, .smi, and .valid files across batches, 
consolidating into a set of 3 files for each epoch

Run:
python combine_generation_batches.py
"""

import os

path_to_folder = "/home/gridsan/dnori/GraphINVENT/output_protac_db/example-job-name/job_0/generation/"
extensions = ['likelihood','smi','valid']


for filename in os.listdir(path_to_folder):
    if 'GEN140' in filename:
        try:
            file_extension = filename[filename.index(".")+1:]
        except:
            file_extension = 'folder'
        if file_extension == 'smi':
            path = path_to_folder + filename
            with open(path, 'r') as input:
                first_instance = filename.index("_")
                filename_sub = filename[first_instance+1:]
                second_instance = filename_sub.index('_')
                epoch_number = filename[first_instance+7:second_instance+1+first_instance]
                
                print('yes')
                print('next file')
                #with open(f"{path_to_folder}aggregated_generation/epoch_{epoch_number}.{file_extension}", 'a+') as f:
                new_path = path_to_folder + "aggregated_generation/epoch_GEN140_AGG.smi"
                with open(new_path, 'a+') as f:
                    for line in input:
                        str_line = str(line)
                        if "SMILES" in str_line:
                            pass
                        else:
                            f.write(line)
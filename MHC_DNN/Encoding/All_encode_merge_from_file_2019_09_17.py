
#%%
# Initiate the Kyte and Doolittle hydrophobicity scale dictionary.
kd_hydrophobicity = { 'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,
       'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I': 4.5,
       'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,
       'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2 }

#%%
# Read tri_propensity file
import csv
binder_propensity = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/A0201_9AA_binder_tripropensity_v2.csv"
propensity_input = csv.reader(open(binder_propensity, "r"))
propensity_dict = {rows[0]:rows[1] for rows in propensity_input}

aa_file = '/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/Encode_dir/1aa_dict.txt'
single_aa_file = csv.reader(open(aa_file, "r"))
dict_aa = {rows[0]:'\t'.join(rows[1:]) for rows in single_aa_file}

aa2_file = '/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/Encode_dir/2aa_dict.txt'
double_aa_file = csv.reader(open(aa2_file, "r"))
dict_2aa = {rows[0]:'\t'.join(rows[1:]) for rows in double_aa_file}

aa3_file = '/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/Encode_dir/3aa_dict.txt'
triple_aa_file = csv.reader(open(aa3_file, "r"))
dict_3aa = {rows[0]:'\t'.join(rows[1:]) for rows in triple_aa_file}


#%%
# Read input and output file
# input_file = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_cluster.txt"
# result = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_cluster_encoded_2019_09_17.txt"

# input_file = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_singleton.txt"
# result = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_singleton_encoded_2019_09_17.txt"

# input_file = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_cluster_extract.txt"
# result = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_cluster_extract_encoded_2019_09_17.txt"

input_file = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_singleton_extract.txt"
result = "/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_singleton_extract_encoded_2019_09_17.txt"


#%%
# Detailed encoding step
from Bio.Data.IUPACData import protein_weights as pw
import re,os
with open(result, 'w', newline='\n') as outfile, open(input_file, 'r', encoding='utf-8') as pep_input:
    for line in pep_input:
        if re.match('#', line):
            continue
        total_aa_weights  = 0
        total_hydrophobicity = 0
        total_propensity = []
        binary_1aa = ''
        binary_2aa = ''
        binary_3aa = ''
        binary_1_2aa = ''
        binary_2_1aa = ''
        binary_1_1_1aa = ''
        # 1 aa, 45 features of each record
        for aa in line.strip():
            total_aa_weights += pw.get(aa.upper(),0)
            total_hydrophobicity += kd_hydrophobicity.get(aa.upper(),0)
            binary_1aa += str(dict_aa.get(aa, 'NA')) + '\t'
        total_aa_weights -= 18*(len(line)-1)
        binary_1aa = binary_1aa[:-1]

        # 2 consecutive aa, 72 features
        for i in range(len(line.strip())-1):
            binary_2aa += str(dict_2aa.get(line[i:i+2],0)) + '\t'
        binary_2aa = binary_2aa[:-1]    

        # 3 consecutive aa, 91 features if binary
        range_list=[] # Contain the following 7 elements
        range_non = 0 # 0
        range_1 = 0 # (0~1)
        range_2 = 0 # [1~2)
        range_3 = 0 # [2~3)
        range_4 = 0 # [3~4)
        range_5 = 0 # [4~5)
        range_6 = 0 # [5~6)
        range_7 = 0 # [6~7)
        range_8 = 0 # [7~8)
        range_9 = 0 # [8~10)
        range_10 = 0 # [10~20)
        range_rest = 0 # rest value
        for i in range(len(line.strip())-2):
            binary_3aa += str(dict_3aa.get(line[i:i+3],0)) + '\t'
            total_propensity.append(propensity_dict.get(line[i:i+3],0))
            each_value = float(propensity_dict.get(line[i:i+3],0))
            if each_value == 0:
                range_non +=1
            elif 0< each_value <1:
                range_1 +=1
            elif 1 <= each_value <2:
                range_2 +=1
            elif 2 <= each_value <3:
                range_3 +=1
            elif 3 <= each_value <4:
                range_4 +=1
            elif 4 <= each_value <5:
                range_5 +=1
            elif 5 <= each_value <6:
                range_6 +=1
            elif 6 <= each_value <7:
                range_7 +=1
            elif 7 <= each_value <8:
                range_8 +=1
            elif 8 <= each_value <10:
                range_9 +=1
            elif 10 <= each_value <20:
                range_10 +=1
            else:
                range_rest += 1
        binary_3aa = binary_3aa[:-1]
        range_list = [range_non,range_1,range_2,range_3,range_4,range_5,range_6,range_7,range_8,range_9,range_10,range_rest]

        # 1-2 aa, 78 binary features
        for i in range(len(line.strip())-3):
            binary_1_2aa += str(dict_3aa.get(str(line[i]+line[i+2:i+4]),0)) + '\t'
        binary_1_2aa = binary_1_2aa[:-1]

        # 2-1 aa, 78 binary features
        for i in range(len(line.strip())-3):
            binary_2_1aa += str(dict_3aa.get(str(line[i:i+2]+line[i+3]),0)) + '\t'
        binary_2_1aa = binary_2_1aa[:-1]

        # 1-1-1 aa, 65 binary features
        for i in range(len(line.strip())-4):
            binary_1_1_1aa += str(dict_3aa.get(str(line[i]+line[i+2]+line[i+4]),0)) + '\t'
        binary_1_1_1aa = binary_1_1_1aa[:-1]
        #print('\t'.join(str(x) for x in range_list))     

        #a total of 450 features output
        # 1 + 1 + 7 + 10 + 45 + 72 + 91 + 78 + 78 + 65 = 450
        total_feature = str(total_aa_weights)+'\t'+str(total_hydrophobicity)+'\t'+str('\t'.join(str(x) for x in total_propensity))+'\t'+str('\t'.join(str(x) for x in range_list))+'\t'+str(binary_1aa)+'\t'+str(binary_2aa)+'\t'+str(binary_3aa)+'\t'+str(binary_1_2aa)+'\t'+str(binary_2_1aa)+'\t'+str(binary_1_1_1aa)

        outfile.write(total_feature + "\n")

#%%

import yaml
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(description='Get forgetting results from yaml files')
parser.add_argument('--path', type=str, default='pt.yaml', 
                    help='path to the result file')
parser.add_argument('--output_path', type=str, default='forgetting.csv', 
help='define output file name')

def read_yaml_file(file_path):
    with open(file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

# Example usage:

args = parser.parse_args()
file_path = args.path
output_path = args.output_path
yaml_data = read_yaml_file(file_path)

len_task = len(yaml_data['history'])

task_history = np.array(yaml_data['history']).squeeze().transpose()

result = []
for i in range(task_history.shape[0]):
    if i == 0:
        result.append(0)
    else:
        res = 0
        for j in range(i + 1):
            res += (np.max(task_history[:, j]) - task_history[i][j])
        res = res / i
        result.append(res)
print(task_history)
result.pop(0)
print(result)
print('average forgetting:',np.mean(result))
# max_acc = np.max(task_history, axis=0)
# last_acc = task_history[:,-1]

# forgetting = max_acc - last_acc
# forgetting = forgetting[:-1]
# # print(forgetting,last_acc)
# avg_forgetting = np.mean(forgetting)
# print('Average forgetting:', avg_forgetting)

# first_row = [i for i in range(1, len_task+1)]
# first_row.insert(0,'Task')
# third_row = []

# with open(output_path, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
    
#     second_row = list(forgetting)
#     second_row.insert(0,'Forgetting')
#     third_row.append('Average Forgetting')
#     third_row.append(np.mean(forgetting))

#     csv_writer.writerow(first_row)
#     csv_writer.writerow(second_row)
#     csv_writer.writerow(third_row)
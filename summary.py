from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def print_f1_score(y_true, y_pred):
    print('f1 score: {0}'.format(f1_score(y_true, y_pred, average='macro')))

def save_result(res, output_path):
    output_file = open(output_path, 'w', encoding='utf-8')
    output_file.write('ID,Expected\n')

    for index in range(len(res)):
        output_file.write('{0},{1}\n'.format(index,res[index]))

    output_file.close()
    print("Result saved to {}\n".format(output_path))

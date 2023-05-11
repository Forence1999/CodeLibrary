'''
This script is used to print the training summary of wav2vec2 model.
Usage:
    python w2v2_train_summary.py [argu1] [argu2]
    argu1: model parameters to be printed, seperated with "," e.g. 'forence_dropout_strategy,forence_dropout_rate'
    argu2: evaluation metrics to be printed, seperated with "," e.g. 'dev_other_best_wer'
'''

from pathlib import Path
import sys


if __name__ == '__main__':
    
    arguments = sys.argv[1:]
    model_parameters = arguments[0].split(',') if len(arguments) > 0 else []
    evaluation_metrics = arguments[1].split(',') if len(arguments) > 1 else []
    
    log_paths = list(Path('./').rglob('*.log'))
    log_paths.sort(key=lambda x: x.parent.name)
    for log_path in log_paths:
        print('-' * 20)
        print(log_path.parent.name.split('_')[0])
        
        with open(log_path, 'r') as file:
            for line in file:
                if '[fairseq.models.wav2vec.wav2vec2_asr]' in line:
                    for argu in model_parameters:
                        if argu in line:
                            idx = line.find(argu)
                            print(line[idx:idx + len(argu) + 10].split(',')[0])
                    break
        
        matching_line = None
        with open(log_path, 'r') as file:
            for line in file:
                if '[dev_other][INFO]' in line:
                    matching_line = line
        
        # 打印最后一行匹配的结果
        if matching_line is not None:
            for argu in evaluation_metrics:
                if argu in matching_line:
                    idx = matching_line.find(argu)
                    print(matching_line[idx:idx + len(argu) + 10].split(',')[0])
        
        print('-' * 20)

import sys

sys.path.append('..')

import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.Ninapro import *
from model.models import *
from loss.loss import *
from runner.runner import *
from metric.metrics import *

import warnings

warnings.filterwarnings("ignore")

'''
Experient 1,
Ninapro DB2,
100ms,150ms,200ms,250ms,300ms
s5,s8,s40
'''


def exp1_run_forone(sub, seq_len):
    # Experimental parameter settings.
    batch_size_pretrain = 1024
    epochs_pretrain = 32
    batch_size = 64
    epochs = 18
    seq_lens = seq_len * 2
    channels = 12
    num_classes = 49
    path_s = f'../data/ninapro/db2_processed/{sub}'
    if not os.path.exists(f'../save_folder_exp2'):
        os.makedirs(f'../save_folder_exp2')
    if not os.path.exists(f'../save_folder_exp2/db2_{seq_len}ms'):
        os.makedirs(f'../save_folder_exp2/db2_{seq_len}ms')

    # Acquire and preprocess the dataset.
    data_train, labels_train, exercise_train, data_val, labels_val, exercise_val = get_data_ninapro_db2(path_s,
                                                                                                        seq_lens)
    input_dim = data_train[0].shape[0]

    # Dataloader for pre-training.
    trainset_1 = Ninapro(data_train, labels_train)
    trainloader_1 = DataLoader(trainset_1, batch_size=batch_size_pretrain, shuffle=True, drop_last=True)

    # Dataloader for formal training.
    trainset_2 = Ninapro_db2(data_train, labels_train, exercise_train)
    valset_2 = Ninapro_db2(data_val, labels_val, exercise_val)
    trainloader_2 = DataLoader(trainset_2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    devloader_2 = DataLoader(valset_2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # The pre-training
    tc_encoder = TC_Encoder(input_dim, channels).to('cuda')
    linear_project = Linear_Project().to('cuda')
    model_pretrain = Model_pretrain(tc_encoder, linear_project)
    save_path_1 = f'../save_folder_exp2/db2_{seq_len}ms/tc_encoder_{sub}.pt'
    gesture_contrastive_loss = Gesture_Contrastive_Loss(temperature=0.012)
    opt_1 = optim.SGD(model_pretrain.parameters(), lr=0.1)
    runner_1 = Runner_v1(model=model_pretrain, loss_fn=gesture_contrastive_loss, optimizer=opt_1, save_path=save_path_1)
    print(f'The pre-training of {sub} starts!')
    runner_1.train(train_loader=trainloader_1, num_epochs=epochs_pretrain, log_stride=64)

    # The formal training
    best_score_log = []
    encoder = model_pretrain.encoder
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True), 1)
    embedding = nn.Embedding(4, 512)
    classifier = Classifier(num_classes=num_classes)
    model = GCS_Former(encoder, decoder, embedding, classifier).to('cuda')
    save_path_2 = f'../save_folder_exp2/db2_{seq_len}ms/model_{sub}.pt'
    metric = Accuracy2()
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    runner = Runner_v3(model=model, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_2)
    print(f'The formal training of {sub} starts!')
    runner.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs, log_stride=128)

    # The 1st round of fine-tuning,lr=0.0001
    runner.load(save_path_2)
    model_finetune1 = GCS_Former_finetune(model.encoder, model.decoder, model.embedding, model.classifier).to('cuda')
    save_path_3 = f'../save_folder_exp2/db2_{seq_len}ms/model_finetune1_{sub}.pt'
    metric_finetune = Accuracy2()
    loss_finetune = nn.CrossEntropyLoss()
    opt_finetune1 = optim.SGD(model_finetune1.parameters(), lr=0.0001)
    runner_finetune1 = Runner_v3(model=model_finetune1, loss_fn=loss_finetune, optimizer=opt_finetune1,
                                 metric=metric_finetune, save_path=save_path_3)
    runner_finetune1.best_score = runner.best_score
    print(f'The 1st round of fine-tuning for {sub} starts!')
    runner_finetune1.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs // 3 * 2,
                           log_stride=128)
    best_score_log.append(runner_finetune1.best_score)

    # The 2nd round of fine-tuning,lr=0.00001
    try:
        runner_finetune1.load(save_path_3)
    except:
        pass

    model_finetune2 = GCS_Former_finetune(model_finetune1.encoder, model_finetune1.decoder, model_finetune1.embedding,
                                          model_finetune1.classifier).to('cuda')

    save_path_4 = f'../save_folder_exp2/db2_{seq_len}ms/model_finetune2_{sub}.pt'

    opt_finetune2 = optim.SGD(model_finetune2.parameters(), lr=0.00001)
    runner_finetune2 = Runner_v3(model=model_finetune2, loss_fn=loss_finetune, optimizer=opt_finetune2,
                                 metric=metric_finetune, save_path=save_path_4)
    runner_finetune2.best_score = runner_finetune1.best_score
    print(f'The 2nd round of fine-tuning for {sub} starts!')
    runner_finetune2.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs // 3 * 2,
                           log_stride=128)
    best_score_log.append(runner_finetune2.best_score)

    # Record the results.
    file_path = f"../save_folder_exp2/db2_best_score.txt"

    if not os.path.exists(file_path):
        open(file_path, 'w').close()
        print("File created successfully.")
    else:
        pass

    file = open(file_path, "a")
    file.write(f"{sub}_{seq_len}ms_best_score: {max(best_score_log)} \n")
    file.close()
    print(f'The training and fine-tuning of {sub} are finished!')
    print('\n')
    return max(best_score_log)


'''
Iterate over each person, then execute the above process sequentially and save the best result of each person.
'''

all_sub = ['DB2_s5', 'DB2_s8', 'DB2_s40']
window_size = [100, 150, 200, 250, 300]
best_scores = []
for sub in all_sub:
    for seq_len in window_size:
        # try:
        best_score = exp1_run_forone(sub, seq_len)
        best_scores.append(best_score)
        # except:
        # print(f"{sub} wrong!")
print("all done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(best_scores)

# all_sub=[ 'DB2_s25', 'DB2_s26', 'DB2_s27', 'DB2_s28', 'DB2_s29', 'DB2_s3', 'DB2_s30', 'DB2_s31', 'DB2_s32',
#           'DB2_s  33', 'DB2_s34', 'DB2_s35', 'DB2_s36', 'DB2_s37', 'DB2_s38', 'DB2_s39', 'DB2_s4', 'DB2_s40', 'DB2_s5',
#           'DB2_s6', 'DB2_s7', 'DB2_s8', 'DB2_s9']

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
Without Gesture Contrastive Loss
'''


def exp1_run_forone(sub):
    # Experimental parameter settings.
    batch_size = 256
    epochs = 16
    seq_lens = 400  # 200ms
    channels = 12
    num_classes = 49
    path_s = f'../data/ninapro/db2_processed/{sub}'

    if not os.path.exists(f'../save_folder_exp4'):
        os.makedirs(f'../save_folder_exp4')
    if not os.path.exists(f'../save_folder_exp4/db2'):
        os.makedirs(f'../save_folder_exp4/db2')

    # Acquire and preprocess the dataset.
    data_train, labels_train, exercise_train, data_val, labels_val, exercise_val = get_data_ninapro_db2(path_s,
                                                                                                        seq_lens)
    input_dim = data_train[0].shape[0]
    print('sample shape:', data_train[0].shape)

    # Dataloader for  training.
    trainset = Ninapro_db2(data_train, labels_train, exercise_train)
    valset = Ninapro_db2(data_val, labels_val, exercise_val)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    devloader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Training
    best_score_log = []
    tc_encoder = TC_Encoder(input_dim, channels).to('cuda')
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True), 1)
    embedding = nn.Embedding(4, 512)
    classifier = Classifier(num_classes=num_classes)
    model_without_gcs = GCS_Former_finetune(tc_encoder, decoder, embedding, classifier).to('cuda')
    save_path = f'../save_folder_exp4/db2/model_without_gcs_{sub}.pt'
    metric = Accuracy2()
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model_without_gcs.parameters(), lr=0.001)
    runner = Runner_v3(model=model_without_gcs, loss_fn=loss, optimizer=opt,
                       metric=metric, save_path=save_path)
    print(f'The training for {sub} starts!')
    runner.train(train_loader=trainloader, dev_loader=devloader, num_epochs=epochs, log_stride=128)
    best_score_log.append(runner.best_score)

    # Record the results.
    file = open("../save_folder_exp4/db2/best_score.txt", "a")
    file.write(f"{sub}_best_score: {max(best_score_log)} \n")
    file.close()
    print(f'The training of {sub} is finished!')
    print('\n')
    return max(best_score_log)


'''
Iterate over each person, then execute the above process sequentially and save the best result of each person.
'''


def exp1_run_forall(all_sub):
    best_scores = []
    for sub in all_sub:
        try:
            best_score = exp1_run_forone(sub)
            best_scores.append(best_score)
        except:
            print(f"{sub} wrong!")
    print("all done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(best_scores)


all_sub = os.listdir('../data/ninapro/db2_processed')
all_sub = ['DB2_s10', 'DB2_s13', 'DB2_s17']
print(all_sub)
exp1_run_forall(all_sub)

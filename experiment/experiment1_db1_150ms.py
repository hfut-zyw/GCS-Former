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
Ninapro DB1,
150ms
'''


def exp1_run_forone(sub):
    # Experimental parameter settings.
    batch_size_pretrain = 2048
    epochs_pretrain = 32
    batch_size = 64
    epochs = 2
    seq_lens = 15
    channels = 10
    num_classes = 52
    path_s = f'../data/ninapro/db1_processed/{sub}'

    if not os.path.exists(f'../save_folder_exp1'):
        os.makedirs(f'../save_folder_exp1')
    if not os.path.exists(f'../save_folder_exp1/db1_150ms'):
        os.makedirs(f'../save_folder_exp1/db1_150ms')

    # Acquire and preprocess the dataset.
    data_train, labels_train, exercise_train, data_val, labels_val, exercise_val = get_data_ninapro_db1(path_s,
                                                                                                        seq_lens)

    # Dataloader for pre-training.
    trainset_1 = Ninapro(data_train, labels_train)
    trainloader_1 = DataLoader(trainset_1, batch_size=batch_size_pretrain, shuffle=True)

    # Dataloader for formal training.
    trainset_2 = Ninapro_db1(data_train, labels_train, exercise_train)
    valset_2 = Ninapro_db1(data_val, labels_val, exercise_val)
    trainloader_2 = DataLoader(trainset_2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    devloader_2 = DataLoader(valset_2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # The pre-training
    tc_encoder = TC_Encoder(seq_lens, channels).to('cuda')
    linear_project = Linear_Project().to('cuda')
    model_pretrain = Model_pretrain(tc_encoder, linear_project)
    save_path_1 = f'../save_folder_exp1/db1_150ms/tc_encoder_{sub}.pt'
    gesture_contrastive_loss = Gesture_Contrastive_Loss(temperature=0.02)
    opt_1 = optim.SGD(model_pretrain.parameters(), lr=0.1)
    runner_1 = Runner_v1(model=model_pretrain, loss_fn=gesture_contrastive_loss, optimizer=opt_1, save_path=save_path_1)
    print(f'The pre-training of {sub} starts!')
    runner_1.train(train_loader=trainloader_1, num_epochs=epochs_pretrain, log_stride=128)

    # The formal training
    best_score_log = []
    encoder = model_pretrain.encoder
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True), 1)
    embedding = nn.Embedding(4, 512)
    classifier = Classifier(num_classes=num_classes)
    model = GCS_Former(encoder, decoder, embedding, classifier).to('cuda')
    save_path_2 = f'../save_folder_exp1/db1_150ms/model_{sub}.pt'
    metric = Accuracy2()
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    runner = Runner_v3(model=model, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_2)
    print(f'The formal training of {sub} starts!')
    runner.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs, log_stride=128)

    # The 1st round of fine-tuning,lr=0.001
    runner.load(save_path_2)
    model_finetune1 = GCS_Former_finetune(model.encoder, model.decoder, model.embedding, model.classifier).to('cuda')
    save_path_3 = f'../save_folder_exp1/db1_150ms/model_finetune1_{sub}.pt'
    metric_finetune = Accuracy2()
    loss_finetune = nn.CrossEntropyLoss()
    opt_finetune1 = optim.SGD(model_finetune1.parameters(), lr=0.001)
    runner_finetune1 = Runner_v3(model=model_finetune1, loss_fn=loss_finetune, optimizer=opt_finetune1,
                                 metric=metric_finetune, save_path=save_path_3)
    runner_finetune1.best_score = runner.best_score
    print(f'The 1st round of fine-tuning for {sub} starts!')
    runner_finetune1.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs, log_stride=128)
    best_score_log.append(runner_finetune1.best_score)

    # The 2nd round of fine-tuning,lr=0.0001
    try:
        runner_finetune1.load(save_path_3)
    except:
        pass

    model_finetune2 = GCS_Former_finetune(model_finetune1.encoder, model_finetune1.decoder, model_finetune1.embedding,
                                          model_finetune1.classifier).to('cuda')

    save_path_4 = f'../save_folder_exp1/db1_150ms/model_finetune2_{sub}.pt'

    opt_finetune2 = optim.SGD(model_finetune2.parameters(), lr=0.0001)
    runner_finetune2 = Runner_v3(model=model_finetune2, loss_fn=loss_finetune, optimizer=opt_finetune2,
                                 metric=metric_finetune, save_path=save_path_4)
    runner_finetune2.best_score = runner_finetune1.best_score
    print(f'The 2nd round of fine-tuning for {sub} starts!')
    runner_finetune2.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs, log_stride=128)
    best_score_log.append(runner_finetune2.best_score)

    # The 3rd round of fine-tuning,lr=0.00001
    try:
        runner_finetune2.load(save_path_4)
    except:
        pass

    model_finetune3 = GCS_Former_finetune(model_finetune2.encoder, model_finetune2.decoder, model_finetune2.embedding,
                                          model_finetune2.classifier).to('cuda')
    save_path_5 = f'../save_folder_exp1/db1_150ms/model_finetune3_{sub}.pt'

    opt_finetune3 = optim.SGD(model_finetune3.parameters(), lr=0.00001)
    runner_finetune3 = Runner_v3(model=model_finetune3, loss_fn=loss_finetune, optimizer=opt_finetune3,
                                 metric=metric_finetune, save_path=save_path_5)
    runner_finetune3.best_score = runner_finetune2.best_score
    print(f'The 3rd round of fine-tuning for {sub} starts!')
    runner_finetune3.train(train_loader=trainloader_2, dev_loader=devloader_2, num_epochs=epochs, log_stride=128)
    best_score_log.append(runner_finetune3.best_score)

    # Record the results.
    file = open("../save_folder_exp1/db1_150ms/best_score.txt", "a")
    file.write(f"{sub}_best_score: {max(best_score_log)} \n")
    file.close()
    print(f'The training and fine-tuning of {sub} are finished!')
    print('\n')
    return max(best_score_log)


'''
Iterate over each person, then execute the above process sequentially and save the best result of each person.
'''


def exp1_run_forall(all_sub):
    best_scores = []
    for sub in all_sub:
        # try:
        best_score = exp1_run_forone(sub)
        best_scores.append(best_score)
        # except:
        # print(f"{sub} wrong!")
    print("all done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(best_scores)


all_sub = []
for i in range(1, 28):
    all_sub.append(f's{i}')
print(all_sub)
exp1_run_forall(all_sub)

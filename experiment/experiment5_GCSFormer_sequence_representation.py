from torch import optim
from torch.utils.data import DataLoader

from dataloader.OH_Sentence import *
from loss.loss import *
from metric.metrics import *
from model.models import *
from runner.runner import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Experient 5,
OH-Sentence DB1,
GCS-Former,
sequence representation
'''


def exp5_run_forone_1(path, sub):
    path_s = os.path.join(path, sub)
    window_size = 150
    channels = 8
    t = 0.06
    sub = sub.split("_")[0]
    best_score_log = []

    if not os.path.exists(f'../save_folder_exp5'):
        os.makedirs(f'../save_folder_exp5')
    if not os.path.exists(f'../save_folder_exp5/sequence_represent'):
        os.makedirs(f'../save_folder_exp5/sequence_represent')
    if not os.path.exists(f'../save_folder_exp5/sequence_represent/{sub}'):
        os.makedirs(f'../save_folder_exp5/sequence_represent/{sub}')

    train_data, train_labels, test_data, test_labels = make_data_oh(path_s, window_size, window_size)
    print(len(train_data), len(test_data))
    num_classes = np.unique(train_labels).shape[0]
    pre_trainset = OH_Dataset(train_data, train_labels)
    pre_trainloader = DataLoader(pre_trainset, shuffle=True, batch_size=50, drop_last=False, collate_fn=collate_fn1)
    trainset = OH_Dataset(train_data, train_labels)
    testset = OH_Dataset(test_data, test_labels)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=50, drop_last=False, collate_fn=collate_fn2)
    devloader = DataLoader(testset, shuffle=False, batch_size=50, collate_fn=collate_fn2)

    # pre-train
    tc_encoder = TC_Encoder(window_size, channels).to('cuda')
    linear_project = Linear_Project().to('cuda')
    model_oh_pretrain = GCS_Former_OH_Pretrain(tc_encoder, linear_project)
    save_path_0 = f'../save_folder_exp5/sequence_represent/{sub}/tc_encoder_{sub}.pt'
    gesture_contrastive_loss = Gesture_Contrastive_Loss(temperature=t)
    opt = optim.SGD(model_oh_pretrain.parameters(), lr=0.1)
    runner0 = Runner_v5_1(model=model_oh_pretrain, loss_fn=gesture_contrastive_loss, optimizer=opt,
                          save_path=save_path_0)
    print(f'The pre-training of {sub} starts!')
    runner0.train(train_loader=pre_trainloader, num_epochs=32, log_stride=16)

    # formal-train
    runner0.load(save_path_0)
    encoder = model_oh_pretrain.encoder
    linear_project = model_oh_pretrain.linear_project
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=128, nhead=8, batch_first=True), 1)
    embedding = nn.Embedding(1, 128)
    classifier = Classifier(128, num_classes)
    model_oh_stopgrad = GCS_Former_OH_stopgrad(encoder, linear_project, decoder, embedding, classifier).to('cuda')
    save_path_1 = f'../save_folder_exp5/sequence_represent/{sub}/model_oh_{sub}.pt'
    metric = Accuracy()
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model_oh_stopgrad.parameters(), lr=0.0001)
    runner1 = Runner_v5_2(model=model_oh_stopgrad, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_1)
    print(f'The formal training of {sub} starts!')
    runner1.train(train_loader=trainloader, dev_loader=devloader, num_epochs=256, log_stride=32)
    best_score_log.append(runner1.best_score)

    # # batch size set 40
    # trainloader = DataLoader(trainset, shuffle=True, batch_size=40, drop_last=False, collate_fn=collate_fn2)
    # devloader = DataLoader(testset, shuffle=False, batch_size=40, collate_fn=collate_fn2)
    #
    # # funetune 1
    # runner1.load(save_path_1)
    # model_oh_finetune1 = GCS_Former_OH(runner1.model.encoder, runner1.model.linear_project, runner1.model.decoder,
    #                                    runner1.model.embedding, runner1.model.classifier).to('cuda')
    # save_path_2 = f'../save_folder_exp5/sequence_represent/{sub}/model_oh_{sub}.pt'
    # metric = Accuracy()
    # loss = nn.CrossEntropyLoss()
    # opt = optim.SGD(model_oh_finetune1.parameters(), lr=0.001)
    # runner2 = Runner_v5_2(model=model_oh_finetune1, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_2)
    # runner2.best_score=runner1.best_score
    # print(f'The finetune 1 of {sub} starts!')
    # runner2.train(train_loader=trainloader, dev_loader=devloader, num_epochs=16, log_stride=32)
    # best_score_log.append(runner2.best_score)
    #
    # # finetune 2
    # try:
    #     runner2.load(save_path_2)
    #     model_oh_finetune2 = GCS_Former_OH(runner2.model.encoder, runner2.model.linear_project, runner2.model.decoder,
    #                                        runner2.model.embedding, runner2.model.classifier).to('cuda')
    # except:
    #     runner1.load(save_path_1)
    #     model_oh_finetune2 = GCS_Former_OH(runner1.model.encoder, runner1.model.linear_project, runner1.model.decoder,
    #                                        runner1.model.embedding, runner1.model.classifier).to('cuda')
    # save_path_3 = f'../save_folder_exp5/sequence_represent/{sub}/model_oh_finetune_{sub}.pt'
    # metric = Accuracy()
    # loss = nn.CrossEntropyLoss()
    # opt = optim.SGD(model_oh_finetune2.parameters(), lr=0.0006)
    # runner3 = Runner_v5_2(model=model_oh_finetune2, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_3)
    # runner3.best_score = runner2.best_score
    # print(f'The finetune 2 of {sub} starts!')
    # runner3.train(train_loader=trainloader, dev_loader=devloader, num_epochs=16, log_stride=32)
    # best_score_log.append(runner3.best_score)
    #
    # # finetune 3
    # try:
    #     runner3.load(save_path_3)
    #     model_oh_finetune4 = GCS_Former_OH(runner3.model.encoder, runner3.model.linear_project, runner3.model.decoder,
    #                                        runner3.model.embedding, runner3.model.classifier).to('cuda')
    # except:
    #     runner2.load(save_path_2)
    #     model_oh_finetune4 = GCS_Former_OH(runner2.model.encoder, runner2.model.linear_project, runner2.model.decoder,
    #                                        runner2.model.embedding, runner2.model.classifier).to('cuda')
    # save_path_4 = f'../save_folder_exp5/sequence_represent/{sub}/model_oh_finetune_{sub}.pt'
    # metric = Accuracy()
    # loss = nn.CrossEntropyLoss()
    # opt = optim.SGD(model_oh_finetune4.parameters(), lr=0.0001)
    # runner4 = Runner_v5_2(model=model_oh_finetune4, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_4)
    # runner4.best_score = runner3.best_score
    # print(f'The finetune 3 of {sub} starts!')
    # runner4.train(train_loader=trainloader, dev_loader=devloader, num_epochs=8, log_stride=32)
    # best_score_log.append(runner3.best_score)

    # Record the results.
    file_path = f"../save_folder_exp5/best_score_log.txt"

    if not os.path.exists(file_path):
        open(file_path, 'w').close()
        print("File created successfully.")
    else:
        pass

    file = open(file_path, "a")
    file.write(f"{sub}_best_score: {max(best_score_log)} \n")
    file.close()
    print(f'The training of {sub} are finished!')
    print('\n')
    return max(best_score_log)


'''
sequence representation
'''


def exp5_run_forall_1(path, all_sub):
    best_scores = []
    for sub in all_sub:
        try:
            best_score = exp5_run_forone_1(path, sub)
            best_scores.append(best_score)
        except:
            print("something wrong!")
    print("all done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(best_scores)


DB1a_path = '../data/oh_sentence/DB1a'
DB1a_subs = ['sub1_邓大柱', 'sub2_王业梅', 'sub3_杨艳', 'sub4_赵鑫', 'sub5_宋雪霞', 'sub6_宋鹏',
             'sub7_李霜', 'sub8_高琴', 'sub9_陈玉茹', 'sub10_张澜', 'sub11_张雷', 'sub12_陈玖玲',
             'sub13_魏明月', 'sub14_陈畅']
DB1b_path = '../data/oh_sentence/DB1b'
DB1b_subs = ['sub15_丁炳权', 'sub16_段浩', 'sub17_汪新宇', 'sub18_王星辰', 'sub19_许金亮']

exp5_run_forall_1(DB1a_path,DB1a_subs)
exp5_run_forall_1(DB1b_path, DB1b_subs)

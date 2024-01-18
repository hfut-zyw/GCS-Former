from torch import optim
from torch.utils.data import DataLoader

from dataloader.OH_Sentence import *
from loss.loss import *
from metric.metrics import *
from model.models import *
from runner.runner import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Experient 6,
OH-Sentence DB1,
75 percent
'''


def exp6_run_forone(path, sub, sub_base):
    sub_base = sub_base.split("_")[0]
    path_s = os.path.join(path, sub)
    window_size = 150
    channels = 8
    t = 0.06
    sub = sub.split("_")[0]
    best_score_log = []

    if not os.path.exists(f'../save_folder_exp6'):
        os.makedirs(f'../save_folder_exp6')
    if not os.path.exists(f'../save_folder_exp6/{sub}_75percent'):
        os.makedirs(f'../save_folder_exp6/{sub}_75percent')

    train_data, train_labels, test_data, test_labels = make_data_oh_75percent(path_s, window_size, window_size)
    print(len(train_data), len(test_data))
    num_classes = np.unique(train_labels).shape[0]
    trainset = OH_Dataset(train_data, train_labels)
    testset = OH_Dataset(test_data, test_labels)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=16, drop_last=False, collate_fn=collate_fn1)
    devloader = DataLoader(testset, shuffle=False, batch_size=16, collate_fn=collate_fn1)

    # load tc-encoder from sub_base
    tc_encoder = TC_Encoder(window_size, channels).to('cuda')
    linear_project = Linear_Project().to('cuda')
    model_oh_pretrain = GCS_Former_OH_Pretrain(tc_encoder, linear_project)
    save_path_0 = f'../save_folder_exp5/average_represent/{sub_base}/tc_encoder_{sub_base}.pt'
    gesture_contrastive_loss = Gesture_Contrastive_Loss(temperature=t)
    opt = optim.SGD(model_oh_pretrain.parameters(), lr=0.1)
    runner0 = Runner_v5_1(model=model_oh_pretrain, loss_fn=gesture_contrastive_loss, optimizer=opt,
                          save_path=save_path_0)

    # train
    runner0.load(save_path_0)
    encoder = model_oh_pretrain.encoder
    decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=512, nhead=2, batch_first=True), 1)
    embedding = nn.Embedding(1, 512)
    classifier = Classifier(512, num_classes)
    model_oh_stopgrad = GCS_Former_OH_stopgrad2(encoder, decoder, embedding, classifier).to('cuda')
    save_path_1 = f'../save_folder_exp6/{sub}_75percent/{sub}_model_oh_{sub_base}.pt'
    metric = Accuracy()
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model_oh_stopgrad.parameters(), lr=0.0001)
    runner1 = Runner_v5_3(model=model_oh_stopgrad, loss_fn=loss, optimizer=opt, metric=metric, save_path=save_path_1)
    print(f'The formal training of {sub} starts!')
    runner1.train(train_loader=trainloader, dev_loader=devloader, num_epochs=200, log_stride=32)
    best_score_log.append(runner1.best_score)

    # funetune
    runner1.load(save_path_1)
    model_oh_finetune1 = GCS_Former_OH2(runner1.model.encoder, runner1.model.decoder,
                                        runner1.model.embedding, runner1.model.classifier).to('cuda')
    metric = Accuracy()
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD(model_oh_finetune1.parameters(), lr=0.00001)
    runner2 = Runner_v5_3(model=model_oh_finetune1, loss_fn=loss, optimizer=opt, metric=metric)
    runner2.best_score=runner1.best_score
    print(f'The finetune of {sub} starts!')
    runner2.train(train_loader=trainloader, dev_loader=devloader, num_epochs=64, log_stride=32)
    best_score_log.append(runner2.best_score)

    # Record the results.
    file_path = f"../save_folder_exp6/{sub}_75percent/best_score_log.txt"

    if not os.path.exists(file_path):
        open(file_path, 'w').close()
        print("File created successfully.")
    else:
        pass

    file = open(file_path, "a")
    file.write(f"TransferLearning_{sub}_best_score_{sub_base}: {max(best_score_log)} \n")
    file.close()
    print(f'The training of {sub} are finished!')
    print('\n')
    return max(best_score_log)


def exp6_run_forall(path, all_sub, all_sub_base):
    best_scores = []
    for sub in all_sub:
        for sub_base in all_sub_base:
            # try:
            best_score = exp6_run_forone(path, sub, sub_base)
            best_scores.append(best_score)
            # except:
                # print("something wrong!")
    print("all done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(best_scores)

all_sub_base=['sub1_邓大柱', 'sub2_王业梅', 'sub3_杨艳', 'sub4_赵鑫', 'sub5_宋雪霞', 'sub6_宋鹏',
             'sub7_李霜', 'sub8_高琴', 'sub9_陈玉茹']

DB1a_path = '../data/oh_sentence/DB1a'
DB1a_subs = [ 'sub10_张澜', 'sub11_张雷', 'sub12_陈玖玲',
             'sub13_魏明月', 'sub14_陈畅']
DB1b_path = '../data/oh_sentence/DB1b'
DB1b_subs = ['sub15_丁炳权', 'sub16_段浩', 'sub17_汪新宇', 'sub18_王星辰', 'sub19_许金亮']

exp6_run_forall(DB1a_path,DB1a_subs,all_sub_base)
exp6_run_forall(DB1b_path, DB1b_subs,all_sub_base)

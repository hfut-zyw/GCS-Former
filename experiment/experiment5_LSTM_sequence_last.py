'''
Experiment 5
Transformer On OH-Sentence DB1
Average accuracy < 20% , overfitting
'''

from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloader.OH_Sentence import *
from metric.metrics import *
from model.models import *
from runner.runner import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Experient 5,
OH-Sentence DB1,
LSTM
'''


def run_for_one(path, sub):
    path_s = os.path.join(path, sub)
    window_size = 150
    stride = 80
    embedding_dim = 128
    segment_D = window_size * 8

    train_data, train_labels, test_data, test_labels = seqmodels_make_data(path_s, window_size, stride)
    print(len(train_data), len(test_data), train_data[0].shape)
    print(len(train_data), len(test_data))
    batch_size_train = len(train_data) // 8
    batch_size_test = len(test_data) // 8
    num_classes = np.unique(train_labels).shape[0]
    if len(train_data) > 1000:
        epochs = 300
    else:
        epochs = 200

    trainset = seqmodels_Dataset(train_data, train_labels)
    testset = seqmodels_Dataset(test_data, test_labels)
    print(trainset[0][0].shape)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size_train, collate_fn=seqmodels_collate_fn1,
                             drop_last=True)
    devloader = DataLoader(testset, shuffle=False, batch_size=batch_size_test, collate_fn=seqmodels_collate_fn1,
                           drop_last=False)

    encoder = nn.LSTM(input_size=1200, hidden_size=128, num_layers=2).to(device)
    classifier = Classifier(embedding_dim, num_classes).to(device)
    lstm_oh = LSTM_OH(encoder, classifier).to(device)
    metric = Accuracy()
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(lstm_oh.parameters(), lr=0.001)
    runner = Runner_v5_3(model=lstm_oh, loss_fn=loss_fn, optimizer=opt, metric=metric)
    runner.train(train_loader=trainloader, dev_loader=devloader, num_epochs=epochs, log_stride=32)
    print(f'best_score_{sub}:', runner.best_score)
    return runner.best_score


def run_for_all(path, subs):
    best_score_log = {}
    for sub in subs:
        best_score_log[sub] = []
        # try:
        score = run_for_one(path, sub)
        best_score_log[sub].append(score)
        # except:
        # pass
    print(best_score_log)


#  Run for DB1a
DB1a_path = '../data/oh_sentence/DB1a'
DB1a_subs = ['sub1_邓大柱', 'sub2_王业梅', 'sub3_杨艳', 'sub4_赵鑫', 'sub5_宋雪霞', 'sub6_宋鹏', 'sub7_李霜',
             'sub8_高琴', 'sub9_陈玉茹',
             'sub10_张澜', 'sub11_张雷', 'sub12_陈玖玲', 'sub13_魏明月', 'sub14_陈畅']
run_for_all(DB1a_path, DB1a_subs)

#  Run for DB1b
DB1b_path = '../data/oh_sentence/DB1b'
DB1b_sub_ = os.listdir(DB1b_path)
DB1b_subs = []
for _ in DB1b_sub_:
    if os.path.isdir(os.path.join(DB1b_path, _)):
        DB1b_subs.append(_)
run_for_all(DB1b_path, DB1b_subs)

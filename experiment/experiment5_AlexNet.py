from torch import optim
from dataloader.OH_Sentence import *
from metric.metrics import *
from model.models import *
from runner.runner import *
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

'''
Experient 5,
OH-Sentence DB1,
Alexnet
'''



def exp5_run_forone_2(path, sub):
    path_s = os.path.join(path, sub)
    window_size = 150
    stride = 150

    train_data, train_labels, test_data, test_labels = make_data_oh(path_s, window_size, stride)
    print(len(train_data), len(test_data))
    batch_size_train = len(train_data) // 8
    batch_size_test = len(test_data) // 8
    if len(train_data) > 1000:
        epochs = 300
    else:
        epochs = 200
    num_classes = np.unique(train_labels).shape[0]

    trainset = OH_Dataset(train_data, train_labels)
    testset = OH_Dataset(test_data, test_labels)
    print(trainset[0][0].shape, trainset[1][0].shape)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size_train, drop_last=False,
                             collate_fn=collate_fn1)
    devloader = DataLoader(testset, shuffle=False, batch_size=batch_size_test, collate_fn=collate_fn1)

    # train
    encoder = AlexNet_Encoder().to(device)
    classifier = Classifier(128, num_classes).to(device)
    vgg16 = AlexNet(encoder, classifier).to(device)

    metric = Accuracy()
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(vgg16.parameters(), lr=0.001)
    runner = Runner_v5_3(model=vgg16, loss_fn=loss, optimizer=opt, metric=metric)
    print(f'The formal training of {sub} starts!')
    runner.train(train_loader=trainloader, dev_loader=devloader, num_epochs=epochs, log_stride=32)

    print(f'{sub}_best_score', runner.best_score)

    return runner.best_score


'''
Iterate over each person, then execute the above process sequentially and save the best result of each person.
'''


# x=torch.randn([4,3,2,3])
# print(x)
# print(x.reshape([12,2,3]))
# print(x.reshape([12,2,3]).reshape(4,3,2,3))
# time.sleep(1000)

def exp5_run_forall(path, all_sub):
    best_scores = []
    for sub in all_sub:
        # try:
        best_score = exp5_run_forone_2(path, sub)
        best_scores.append(best_score)
        # except:
        # print(f"{sub} wrong!")
    print("all done!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(best_scores)


#  Run for DB1a
DB1a_path = '../data/oh_sentence/DB1a'
DB1a_subs = ['sub1_邓大柱', 'sub2_王业梅', 'sub3_杨艳', 'sub4_赵鑫', 'sub5_宋雪霞', 'sub6_宋鹏', 'sub7_李霜',
             'sub8_高琴', 'sub9_陈玉茹',
             'sub10_张澜', 'sub11_张雷', 'sub12_陈玖玲', 'sub13_魏明月', 'sub14_陈畅']
print(DB1a_subs)
exp5_run_forall(DB1a_path, DB1a_subs)

#  Run for DB1b
DB1b_path = '../data/oh_sentence/DB1b'
DB1b_sub_ = os.listdir(DB1b_path)
DB1b_subs = []
for _ in DB1b_sub_:
    if os.path.isdir(os.path.join(DB1b_path, _)):
        DB1b_subs.append(_)
print(DB1b_subs)
exp5_run_forall(DB1b_path, DB1b_subs)

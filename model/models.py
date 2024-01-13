import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TC_Encoder(nn.Module):
    def __init__(self, input_dim,channels):
        super(TC_Encoder, self).__init__()
        self.channels = channels # Ninapro db1 (10) ,
        self.input_dim = input_dim     # Ninapro db1 (15,20) ,
        self.channel_encoder = nn.ModuleList()
        if self.input_dim < 21:
            for idx in range(self.channels):
                self.channel_encoder.append(
                    nn.Sequential(
                        nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0),  # 64xD
                        nn.BatchNorm1d(64)
                        # nn.ReLU(inplace=True)
                        # nn.Dropout(p=0.5)
                    )
                )
        elif self.input_dim < 41:
            for idx in range(self.channels):
                self.channel_encoder.append(
                    nn.Sequential(
                        nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0),  # 64xD
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5)
                    )
                )
        else:
            for idx in range(self.channels):
                self.channel_encoder.append(
                    nn.Sequential(
                        nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),  # 64xD
                        nn.BatchNorm1d(32),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3)
                    )
                )

        self.time_encoder = nn.ModuleList()
        for idx in range(self.input_dim):
            self.time_encoder.append(
                nn.Sequential(
                    nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(4),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(4),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(4, 8, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm1d(8),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(8, 8, kernel_size=1, stride=1, padding=0),  # 64xD
                    nn.BatchNorm1d(8)
                    # nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.1)
                )
            )
        if self.input_dim < 21:
            concat_dim = 64 * input_dim * self.channels + 8 * input_dim * self.channels
        elif self.input_dim < 41:
            concat_dim = 64 * ((input_dim-1)//2//2+1) * self.channels
        else:
            concat_dim = 32 * input_dim * self.channels
        print(concat_dim)
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )


    def forward(self, x):
        branch1 = []
        for idx in range(self.channels):
            branch1.append(self.channel_encoder[idx](x[:, :, idx, :]).view(x.shape[0], -1))
        branch2 = []
        if self.input_dim < 21:
            for idx in range(self.input_dim):
                branch2.append(self.time_encoder[idx](x[:, :, :, idx]).view(x.shape[0], -1))
        else:
            pass
        branch = branch1 + branch2
        output = torch.cat(branch, dim=1)
        # print(output.shape)
        output = self.fc(output)
        return output


class Linear_Project(nn.Module):
    def __init__(self,embed_dim=512,num_classes=52):
        super(Linear_Project,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.fc(x)

class Model_pretrain(nn.Module):
    def __init__(self, encoder, linear_project):
        super(Model_pretrain, self).__init__()
        self.encoder = encoder
        self.linear_project=linear_project

    def forward(self, x):
        x = self.encoder(x)
        x=self.linear_project(x)
        return x

class Classifier(nn.Module):
    def __init__(self,embed_dim=512,num_classes=None):
        super(Classifier,self).__init__()
        num_classes = num_classes + 3     # gestures + exercises
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(128, num_classes)
    def forward(self,x):
        output = self.fc1(x)
        output = self.fc2(output)
        output = self.output_layer(output)
        return output

class GCS_Former(nn.Module):
    def __init__(self,encoder,decoder,embedding,classifier):
        super(GCS_Former, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.embedding=embedding
        self.classifier=classifier
    def forward(self,src,tgt,tgt_mask):
        memory=self.encoder(src).unsqueeze(1).detach() # Bx1x512
        tgt=self.embedding(tgt)               # Bx2x512
        out=self.decoder(tgt,memory,tgt_mask=tgt_mask) # Bx2x512
        out=self.classifier(out)              # Bx2xnum_classes
        return out


class GCS_Former_finetune(nn.Module):
    def __init__(self,encoder,decoder,embedding,classifier):
        super(GCS_Former_finetune, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.embedding=embedding
        self.classifier=classifier
    def forward(self,src,tgt,tgt_mask):
        memory=self.encoder(src).unsqueeze(1) # Bx1x512
        tgt=self.embedding(tgt)               # Bx2x512
        out=self.decoder(tgt,memory,tgt_mask=tgt_mask) # Bx2x512
        out=self.classifier(out)              # Bx2xnum_classes
        return out


class GCS_Former_OH_Pretrain(nn.Module):
    def __init__(self, encoder, linear_project):
        super(GCS_Former_OH_Pretrain, self).__init__()
        self.encoder = encoder
        self.linear_project = linear_project

    def forward(self, x, lens, mask):
        B = x.shape[0]
        L = x.shape[1]
        c = x.shape[2]
        d = x.shape[3]

        x = x.reshape([B * L, c, d])  # Blx8x80
        x = x.unsqueeze(1)  # Blx1x8x80
        x = self.encoder(x)  # Blx512
        x = x.reshape([B, L, 512])  # BxLx512
        x = x.masked_fill(mask, 0)  # BxLx512
        x = x.sum(dim=1)  # Bx512

        lens = lens.unsqueeze(-1)  # Bx1
        x = x / lens  # Bx512
        x = self.linear_project(x)  # Bx128
        return x


class GCS_Former_OH_stopgrad(nn.Module):
    def __init__(self, encoder, linear_project, decoder, embedding, classifier):
        super(GCS_Former_OH_stopgrad, self).__init__()
        self.encoder = encoder
        self.linear_project = linear_project
        self.decoder = decoder
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, src, src_key_padding_mask):
        B = src.shape[0]
        L = src.shape[1]
        c = src.shape[2]
        d = src.shape[3]

        src = src.reshape([B * L, c, d])  # Blx8x80
        src = src.unsqueeze(1)  # Blx1x8x80
        src = self.encoder(src)  # Blx512
        src = self.linear_project(src)  # BLx128
        memory = src.reshape([B, L, 128]).detach()  # BxLx128

        tgt = torch.zeros([B, 1]).long().to(device)  # Bx1
        tgt = self.embedding(tgt)  # Bx1x128
        out = self.decoder(tgt, memory, memory_key_padding_mask=src_key_padding_mask)  # Bx1x128
        out = out.squeeze(1)  # Bx128
        out = self.classifier(out)  # Bxnum_classes
        return out


class GCS_Former_OH(nn.Module):
    def __init__(self, encoder, linear_project, decoder, embedding, classifier):
        super(GCS_Former_OH, self).__init__()
        self.encoder = encoder
        self.linear_project = linear_project
        self.decoder = decoder
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, src, src_key_padding_mask):
        B = src.shape[0]
        L = src.shape[1]
        c = src.shape[2]
        d = src.shape[3]

        src = src.reshape([B * L, c, d])  # Blx8x80
        src = src.unsqueeze(1)  # Blx1x8x80
        src = self.encoder(src)  # Blx512
        src = self.linear_project(src)  # BLx128
        memory = src.reshape([B, L, 128])  # BxLx128

        tgt = torch.zeros([B, 1]).long().to(device)  # Bx1
        tgt = self.embedding(tgt)  # Bx1x128
        out = self.decoder(tgt, memory, memory_key_padding_mask=src_key_padding_mask)  # Bx1x128
        out = out.squeeze(1)  # Bx128
        out = self.classifier(out)  # Bxnum_classes
        return out

###

class GCS_Former_OH_stopgrad2(nn.Module):
    def __init__(self, encoder,  decoder, embedding, classifier):
        super(GCS_Former_OH_stopgrad2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, x, lens, mask):
        B = x.shape[0]
        L = x.shape[1]
        c = x.shape[2]
        d = x.shape[3]

        x = x.reshape([B * L, c, d])  # Blx8x150
        x = x.unsqueeze(1)  # Blx1x8x150
        x = self.encoder(x)  # Blx512
        x = x.reshape([B, L, 512])  # BxLx512
        x = x.masked_fill(mask, 0)  # BxLx512
        x = x.sum(dim=1)  # Bx512

        lens = lens.unsqueeze(-1)  # Bx1
        memory = (x / lens).unsqueeze(1).detach()  # Bx1x512

        tgt = torch.zeros([B, 1]).long().to(device)  # Bx1
        tgt = self.embedding(tgt)  # Bx1x512
        out = self.decoder(tgt, memory)  # Bx1x512
        out = out.squeeze(1)  # Bx512
        out = self.classifier(out)  # Bxnum_classes
        return out

class GCS_Former_OH2(nn.Module):
    def __init__(self, encoder, decoder, embedding, classifier):
        super(GCS_Former_OH2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, x, lens, mask):
        B = x.shape[0]
        L = x.shape[1]
        c = x.shape[2]
        d = x.shape[3]

        x = x.reshape([B * L, c, d])  # Blx8x150
        x = x.unsqueeze(1)  # Blx1x8x150
        x = self.encoder(x)  # Blx512
        x = x.reshape([B, L, 512])  # BxLx512
        x = x.masked_fill(mask, 0)  # BxLx512
        x = x.sum(dim=1)  # Bx512

        lens = lens.unsqueeze(-1)  # Bx1
        memory = (x / lens).unsqueeze(1).detach()  # Bx1x512

        tgt = torch.zeros([B, 1]).long().to(device)  # Bx1
        tgt = self.embedding(tgt)  # Bx1x512
        out = self.decoder(tgt, memory)  # Bx1x512
        out = out.squeeze(1)  # Bx512
        out = self.classifier(out)  # Bxnum_classes
        return out


class Transformer_OH(nn.Module):
    def __init__(self, linear, encoder, decoder, embedding, classifier):
        super(Transformer_OH, self).__init__()
        self.linear = linear
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, src,src_key_padding_mask):
        B = src.shape[0]
        L = src.shape[1]
        D = src.shape[2]

        src = self.linear(src)  # BxLx128
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # BxLx128
        # memory = memory[:,0,:].unsqueeze(1)

        tgt = torch.zeros([B, 1]).long().to(device)  # Bx1
        tgt = self.embedding(tgt)  # Bx1x128
        out = self.decoder(tgt, memory, memory_key_padding_mask=src_key_padding_mask)  # Bx1x128
        #  = self.decoder(tgt, memory)  # Bx1x128
        out = out.squeeze(1)  # Bx128
        out = self.classifier(out)  # Bxnum_classes

        return out

class Transformer_OH2(nn.Module):
    def __init__(self, linear, encoder, decoder, embedding, classifier):
        super(Transformer_OH2, self).__init__()
        self.linear = linear
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.classifier = classifier

    def forward(self, x, lens, mask):
        B = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]
        src_key_padding_mask=mask.squeeze(-1)

        x = self.linear(x)  # BxLx128
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # BxLx128
        x = x.masked_fill(mask, 0)  # BxLx128
        x = x.sum(dim=1)  # Bx128
        lens = lens.unsqueeze(-1)  # Bx1
        memory = (x / lens).unsqueeze(1)  # Bx1x128

        tgt = torch.zeros([B, 1]).long().to(device)  # Bx1
        tgt = self.embedding(tgt)  # Bx1x128
        out = self.decoder(tgt, memory)  # Bx1x128
        #  = self.decoder(tgt, memory)  # Bx1x128
        out = out.squeeze(1)  # Bx128
        out = self.classifier(out)  # Bxnum_classes

        return out


##
class AlexNet_exp4(nn.Module):
    def __init__(self, input_size, embed_dim=512):
        super(AlexNet_exp4, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * input_size, 1280)
        self.fc2 = nn.Linear(1280, embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape((B, 1, -1))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x

class AlexNet_exp4_2(nn.Module):
    def __init__(self, input_size, embed_dim=512):
        super(AlexNet_exp4_2, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * input_size, 1280)
        self.fc2 = nn.Linear(1280, embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape((B, 1, -1))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x

class AlexNet_Encoder(nn.Module):
    def __init__(self):
        super(AlexNet_Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 18 * 1, 256)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x, lens, padding_mask):
        x = x.transpose(-1, -2)
        B = x.shape[0]
        L = x.shape[1]
        d = x.shape[2]
        c = x.shape[3]

        x = x.reshape([B * L, d, c])  # BLx150x8
        x = x.unsqueeze(1)  # BLx1x150x8

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))
        x = x.view(-1, 128 * 18 * 1)
        x = self.relu6(self.fc1(x))
        x = self.fc2(x)

        x = x.reshape([B, L, 128])  # BxLx128
        x = x.masked_fill(padding_mask, 0.0)
        x = x.sum(dim=1)  # Bx128
        lens = lens.unsqueeze(-1)  # Bx1
        x = x / lens  # Bx128

        return x


class AlexNet(nn.Module):
    def __init__(self, encoder, classifier):
        super(AlexNet, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, lens, mask):
        x = self.encoder(x, lens, mask)
        x = self.classifier(x)
        return x

class LSTM_OH(nn.Module):
    def __init__(self, encoder, classifier):
        super(LSTM_OH, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, lens, mask):
        B = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]

        x, (_, _) = self.encoder(x)  # BxLx128
        idx = (lens - 1).unsqueeze(1).unsqueeze(2).expand(x.size(0), 1, x.size(2))
        out = x.gather(1, idx).squeeze(1)  # Bx128
        out = self.classifier(out)  # Bxnum_classes

        return out

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18_Encoder(nn.Module):
    def __init__(self, block, layers, num_classes=128):
        super(ResNet18_Encoder, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, lens, padding_mask):
        B = x.shape[0]
        L = x.shape[1]
        c = x.shape[2]
        d = x.shape[3]

        x = x.reshape([B * L, c, d])  # BLx8x150
        x = x.unsqueeze(1)  # BLx1x8x150

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.reshape([B, L, 128])  # BxLx128
        x = x.masked_fill(padding_mask, 0.0)
        x = x.sum(dim=1)  # Bx128
        lens = lens.unsqueeze(-1)  # Bx1
        x = x / lens  # Bx128

        return x


def ResNet():
    return ResNet18_Encoder(ResidualBlock, [2, 2, 2, 2])


class ResNet18(nn.Module):
    def __init__(self, encoder, classifier):
        super(ResNet18, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, lens, mask):
        x = self.encoder(x, lens, mask)
        x = self.classifier(x)
        return x
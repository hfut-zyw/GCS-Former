import matplotlib.pyplot as plt
import torch


class Runner_v0():
    def __init__(self, model=None, loss_fn=None, optimizer=None, metric=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.save_path = save_path
        self.train_epoch_loss = []
        self.train_step_loss = []
        self.dev_stride_score = []
        self.dev_stride_loss = []
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, num_epochs=1, log_stride=1):
        step = 0
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in train_loader:
                self.model.train()
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                self.train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % log_stride == 0 or step == total_steps - 1:
                    dev_score, dev_loss = self.evaluate(dev_loader)
                    self.dev_stride_score.append((step, dev_score))
                    self.dev_stride_loss.append((step, dev_loss))
                    print("[Evaluate]  score:{:.4f} loss:{:.4f}".format(dev_score, dev_loss))
                    if dev_score > self.best_score:
                        self.best_score = dev_score
                        if self.save_path:
                            self.save(self.save_path)
                step += 1
            self.train_epoch_loss.append(total_loss / len(train_loader))
        # self.plot()

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        self.metric.reset()
        total_devloss = 0
        for x, y in dev_loader:
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            total_devloss += loss.item()
            self.metric.update(logits, y)
        dev_loss = total_devloss / len(dev_loader)
        dev_score = self.metric.accumulate()
        return dev_score, dev_loss

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.cla()
        sample_step = 20
        plt.subplot(2, 1, 1)
        plt.xlabel("step")
        plt.ylabel("loss")
        train_items = self.train_step_loss[::sample_step]
        train_x = [x[0] for x in train_items]
        train_loss = [x[1] for x in train_items]
        plt.plot(train_x, train_loss, label="Train loss")
        plt.plot(*zip(*self.dev_stride_loss), label="Dev loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.xlabel("step")
        plt.ylabel("score")
        plt.plot(*zip(*self.dev_stride_score), label="Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.draw()

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        logits = self.model(x)
        preds = logits.argmax()
        return preds

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Best model has been updated and saved!", "\n")

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")


'''
Input : x, y
'''


class Runner_v1():
    def __init__(self, model=None, loss_fn=None, optimizer=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_path = save_path

    def train(self, train_loader, num_epochs=1, log_stride=1):
        step = 0
        train_step_loss = []
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in train_loader:
                self.model.train()
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                    if self.save_path:
                        self.save(self.save_path)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
        print("done!")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")


'''
Input : x, y, key_padding_mask
'''


class Runner_v2():
    def __init__(self, model=None, loss_fn=None, optimizer=None, metric=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.save_path = save_path
        self.train_epoch_loss = []
        self.train_step_loss = []
        self.dev_stride_score = []
        self.dev_stride_loss = []
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, num_epochs=1, log_stride=1):
        step = 0
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y, key_padding_mask in train_loader:
                self.model.train()
                logits = self.model(x, key_padding_mask)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                self.train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % log_stride == 0 or step == total_steps - 1:
                    dev_score, dev_loss = self.evaluate(dev_loader)
                    self.dev_stride_score.append((step, dev_score))
                    self.dev_stride_loss.append((step, dev_loss))
                    print("[Evaluate]  score:{:.4f} loss:{:.4f}".format(dev_score, dev_loss))
                    if dev_score > self.best_score:
                        self.best_score = dev_score
                        if self.save_path:
                            self.save(self.save_path)
                step += 1
            self.train_epoch_loss.append(total_loss / len(train_loader))
        # self.plot()

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        self.metric.reset()
        total_devloss = 0
        for x, y, mask in dev_loader:
            predicts = self.model(x, mask)
            loss = self.loss_fn(predicts, y)
            total_devloss += loss.item()
            self.metric.update(predicts, y)
        dev_loss = total_devloss / len(dev_loader)
        dev_score = self.metric.accumulate()
        return dev_score, dev_loss

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.cla()
        sample_step = 20
        plt.subplot(2, 1, 1)
        plt.xlabel("step")
        plt.ylabel("loss")
        train_items = self.train_step_loss[::sample_step]
        train_x = [x[0] for x in train_items]
        train_loss = [x[1] for x in train_items]
        plt.plot(train_x, train_loss, label="Train loss")
        plt.plot(*zip(*self.dev_stride_loss), label="Dev loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.xlabel("step")
        plt.ylabel("score")
        plt.plot(*zip(*self.dev_stride_score), label="Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.draw()

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Best model has been updated and saved!", "\n")


'''
Input :  src, tgt, tgt_mask, labels
'''


class Runner_v3():
    def __init__(self, model=None, loss_fn=None, optimizer=None, metric=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.save_path = save_path
        self.train_epoch_loss = []
        self.train_step_loss = []
        self.dev_stride_score = []
        self.dev_stride_loss = []
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, num_epochs=1, log_stride=1):
        step = 0
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for src, tgt, tgt_mask, labels in train_loader:
                self.model.train()
                logits = self.model(src, tgt, tgt_mask)  # BxLxC
                loss = self.loss_fn(logits.transpose(-1, -2), labels)
                total_loss += loss.item()
                self.train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % log_stride == 0 or step == total_steps - 1:
                    dev_score, dev_loss = self.evaluate(dev_loader)
                    self.dev_stride_score.append((step, dev_score))
                    self.dev_stride_loss.append((step, dev_loss))
                    print("[Evaluate]  score:{:.4f} loss:{:.4f}".format(dev_score, dev_loss))
                    if dev_score > self.best_score:
                        self.best_score = dev_score
                        if self.save_path:
                            self.save(self.save_path)
                step += 1
            self.train_epoch_loss.append(total_loss / len(train_loader))
        # self.plot()

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        self.metric.reset()
        total_devloss = 0
        for src, tgt, tgt_mask, labels in dev_loader:
            logits = self.model(src, tgt, tgt_mask)  # BxLxC
            loss = self.loss_fn(logits.transpose(-1, -2), labels)
            total_devloss += loss.item()
            self.metric.update(logits, labels)
        dev_loss = total_devloss / len(dev_loader)
        dev_score = self.metric.accumulate()
        return dev_score, dev_loss

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.cla()
        sample_step = 20
        plt.subplot(2, 1, 1)
        plt.xlabel("step")
        plt.ylabel("loss")
        train_items = self.train_step_loss[::sample_step]
        train_x = [x[0] for x in train_items]
        train_loss = [x[1] for x in train_items]
        plt.plot(train_x, train_loss, label="Train loss")
        plt.plot(*zip(*self.dev_stride_loss), label="Dev loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.xlabel("step")
        plt.ylabel("score")
        plt.plot(*zip(*self.dev_stride_score), label="Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.draw()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Best model has been updated and saved!", "\n")

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")


'''
Encoder Average & GCS-LOSS Pre-Training
Input : x,y,lens,mask
'''


class Runner_v5_1():
    def __init__(self, model=None, loss_fn=None, optimizer=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_path = save_path

    def train(self, train_loader, num_epochs=1, log_stride=1):
        step = 0
        train_step_loss = []
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y, lens, mask in train_loader:
                self.model.train()
                logits = self.model(x, lens, mask)
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                    if self.save_path:
                        self.save(self.save_path)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
        print("done!")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")


'''
Transformer Decoder Runner
Input : x,y,mask
'''


class Runner_v5_2():
    def __init__(self, model=None, loss_fn=None, optimizer=None, metric=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.save_path = save_path
        self.train_epoch_loss = []
        self.train_step_loss = []
        self.dev_stride_score = []
        self.dev_stride_loss = []
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, num_epochs=1, log_stride=1):
        step = 0
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y, mask in train_loader:
                self.model.train()
                logits = self.model(x, mask)  # BxD
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                self.train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % log_stride == 0 or step == total_steps - 1:
                    dev_score, dev_loss = self.evaluate(dev_loader)
                    self.dev_stride_score.append((step, dev_score))
                    self.dev_stride_loss.append((step, dev_loss))
                    print("[Evaluate]  score:{:.4f} loss:{:.4f}".format(dev_score, dev_loss))
                    if dev_score > self.best_score:
                        self.best_score = dev_score
                        if self.save_path:
                            self.save(self.save_path)
                step += 1
            self.train_epoch_loss.append(total_loss / len(train_loader))
        # self.plot()

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        self.metric.reset()
        total_devloss = 0
        for x, y, mask in dev_loader:
            logits = self.model(x, mask)  # BxD
            loss = self.loss_fn(logits, y)
            total_devloss += loss.item()
            self.metric.update(logits, y)
        dev_loss = total_devloss / len(dev_loader)
        dev_score = self.metric.accumulate()
        return dev_score, dev_loss

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.cla()
        sample_step = 20
        plt.subplot(2, 1, 1)
        plt.xlabel("step")
        plt.ylabel("loss")
        train_items = self.train_step_loss[::sample_step]
        train_x = [x[0] for x in train_items]
        train_loss = [x[1] for x in train_items]
        plt.plot(train_x, train_loss, label="Train loss")
        plt.plot(*zip(*self.dev_stride_loss), label="Dev loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.xlabel("step")
        plt.ylabel("score")
        plt.plot(*zip(*self.dev_stride_score), label="Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.draw()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Best model has been updated and saved!", "\n")

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")


'''
Encoder Average  Runner
Input : x,y,lens,mask
'''


class Runner_v5_3():
    def __init__(self, model=None, loss_fn=None, optimizer=None, metric=None, save_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.save_path = save_path
        self.train_epoch_loss = []
        self.train_step_loss = []
        self.dev_stride_score = []
        self.dev_stride_loss = []
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, num_epochs=1, log_stride=1):
        step = 0
        total_steps = num_epochs * len(train_loader)
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y, lens, mask in train_loader:
                self.model.train()
                logits = self.model(x, lens, mask)  # BxD
                loss = self.loss_fn(logits, y)
                total_loss += loss.item()
                self.train_step_loss.append((step, loss.item()))
                if step % log_stride == 0 or step == total_steps - 1:
                    print("[Train]  epoch:{}/{} step:{}/{} loss:{:.4f}".format(epoch, num_epochs, step, total_steps,
                                                                               loss.item()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % log_stride == 0 or step == total_steps - 1:
                    dev_score, dev_loss = self.evaluate(dev_loader)
                    self.dev_stride_score.append((step, dev_score))
                    self.dev_stride_loss.append((step, dev_loss))
                    print("[Evaluate]  score:{:.4f} loss:{:.4f}".format(dev_score, dev_loss))
                    if dev_score > self.best_score:
                        self.best_score = dev_score
                        if self.save_path:
                            self.save(self.save_path)
                step += 1
            self.train_epoch_loss.append(total_loss / len(train_loader))
        # self.plot()

    @torch.no_grad()
    def evaluate(self, dev_loader):
        self.model.eval()
        self.metric.reset()
        total_devloss = 0
        for x, y, lens, mask in dev_loader:
            logits = self.model(x, lens, mask)  # BxD
            loss = self.loss_fn(logits, y)
            total_devloss += loss.item()
            self.metric.update(logits, y)
        dev_loss = total_devloss / len(dev_loader)
        dev_score = self.metric.accumulate()
        return dev_score, dev_loss

    def plot(self):
        plt.figure(figsize=(8, 8))
        plt.cla()
        sample_step = 20
        plt.subplot(2, 1, 1)
        plt.xlabel("step")
        plt.ylabel("loss")
        train_items = self.train_step_loss[::sample_step]
        train_x = [x[0] for x in train_items]
        train_loss = [x[1] for x in train_items]
        plt.plot(train_x, train_loss, label="Train loss")
        plt.plot(*zip(*self.dev_stride_loss), label="Dev loss")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.xlabel("step")
        plt.ylabel("score")
        plt.plot(*zip(*self.dev_stride_score), label="Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.draw()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print("Best model has been updated and saved!", "\n")

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print("model loaded!")

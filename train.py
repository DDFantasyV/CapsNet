import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix,roc_curve, auc,classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
# from sklearn.svm import SVC


INPUT_SIZE = (1, 28, 28)
transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(INPUT_SIZE[1:], padding=2),
    torchvision.transforms.ToTensor(),
])


def Slidingwindow(dataX, dataY, STEPS=30):
    X = []
    Y = []
    # 序列的第i项和后面的STEPS-1项合在一起作为输入;
    # 第i+STEPS项和后面的PREDICT_STEPS-1项作为输出
    # 即用数据的前STPES个点的信息，预测后面的PREDICT_STEPS个点的值
    for i in range(dataX.shape[0] - STEPS):
        X.append(dataX[i:i + STEPS, :])
        Y.append(dataY[i:i + STEPS])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=int)

filepath = 'traindata_wr.csv'
data = pd.read_csv(filepath, header=None, encoding='utf-8')  ##数据读取
data = data.values  ##转换为array格式
unit_number_redundant = data[:, -1]  # 提取出冗余的unit编号
unit_number = np.unique(unit_number_redundant)  # 删除unit编号中的冗余部分
unit_nums = unit_number.shape[0]  # 故障数
unit_number_list = []
X1 = []
Y1 = []

for i in range(0, unit_nums):
    condition_i = data[:, -1] == i  # 找出对应编号的数据下标集合
    unit_index_i = np.where(condition_i)
    unit_number_i_index = unit_index_i[0]
    #     print(unit_index_i)
    #     print(unit_number_i_index)
    unit_number_i = data[unit_number_i_index, :]
    #     print(unit_number_i)
    dataX = unit_number_i[:, 0:data.shape[1] - 1]
    dataY = unit_number_i[:, -1]
    # print(dataX) # Y为标签 X为训练数据
    # dataX=preprocessing.scale(dataX,axis=0)##Z-score归一化,0-行，1-列
    dataX, dataY = Slidingwindow(dataX, dataY, 40)  ##滑动时窗
    # print("1", dataX.shape)
    X1.append(dataX)
    # print("2", len(X1))
    Y1.append(dataY)
# print("2", len(X1))
dataX1 = X1[0]
# print(X1[2])
# print(type(dataX1))
dataY1 = Y1[0]
# print(dataY1)

for i in range(1, len(X1)):
    dataXx = X1[i]
    dataYx = Y1[i]
    # print('Y!',Y1[i])
    dataX1 = np.append(dataX1, dataXx, axis=0)
    dataY1 = np.append(dataY1, dataYx, axis=0)

dataX = dataX1
dataY = dataY1
# print(dataX.shape)
# print(dataY.shape)
dataY = dataY[:, 39]
dataY2 = []

for i in range(len(dataY)):
    dataY_one_hot = np.zeros(unit_nums)
    dataY_one_hot[dataY[i]] = 1
    dataY2.append(dataY_one_hot)
dataY = np.array(dataY2, dtype=np.float32)
# print(dataY.shape)
# print(dataY)
X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.8, random_state=0)  # 随机划分样本数据为训练集和测试集
X_train1, X_test, Y_train1, Y_test = train_test_split(X_test, Y_test, test_size=0.3, random_state=0)  # 随机划分样本数据为训练集和测试集
# print("** Training data: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

trn_dataset = X_train
tst_dataset = X_test
# trn_dataset = torchvision.datasets.MNIST('.', train=True, download=True, transform=transforms)
# tst_dataset = torchvision.datasets.MNIST('.', train=False, download=True, transform=transforms)
# print('Images for training: %d' % len(trn_dataset))
# print('Images for testing: %d' % len(tst_dataset))

BATCH_SIZE = 128 # Batch size not specified in the paper
trn_loader = torch.utils.data.DataLoader(trn_dataset, BATCH_SIZE, shuffle=True)
tst_loader = torch.utils.data.DataLoader(tst_dataset, BATCH_SIZE, shuffle=False)
# print(trn_loader)

print(X_train)
print(X_train.shape)
print(Y_train)
print(Y_train.shape)

# Define CapsNet
class Conv1(torch.nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_size=9):
        super(Conv1, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activation = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

# Primary Capsules
class PrimaryCapsules(torch.nn.Module):
    def __init__(self, input_shape=(256, 20, 20), capsule_dim=8,
                 out_channels=32, kernel_size=9, stride=2):
        super(PrimaryCapsules, self).__init__()
        self.input_shape = input_shape
        self.capsule_dim = capsule_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = self.input_shape[0]

        self.conv = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels * self.capsule_dim,
            self.kernel_size,
            self.stride
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.permute(0, 2, 3, 1).contiguous()  # permute调换tensor中的维度，contiguous断开数据前后内存关系，进行深拷贝
        # print(x.shape)
        x = x.view(-1, x.size()[1], x.size()[2], self.out_channels, self.capsule_dim)  # batch* n1 * n2 * 32* 8
        # print(x.shape)
        return x

# Routing
class Routing(torch.nn.Module):
    def __init__(self, caps_dim_before=8, caps_dim_after=16,
                 n_capsules_before=(6 * 6 * 32), n_capsules_after=10):
        super(Routing, self).__init__()
        self.n_capsules_before = n_capsules_before  # 胶囊层前为6*6*32，变换为10
        self.n_capsules_after = n_capsules_after
        self.caps_dim_before = caps_dim_before  # 维度前为8，变化为16
        self.caps_dim_after = caps_dim_after

        # Parameter initialization not specified in the paper
        n_in = self.n_capsules_before * self.caps_dim_before
        variance = 2 / (n_in)
        std = np.sqrt(variance)
        self.W = torch.nn.Parameter(
            torch.randn(
                self.n_capsules_before,
                self.n_capsules_after,
                self.caps_dim_after,
                self.caps_dim_before) * std,
            requires_grad=True)

    # Equation (1)
    @staticmethod
    def squash(s):
        s_norm = torch.norm(s, p=2, dim=-1, keepdim=True)
        s_norm2 = torch.pow(s_norm, 2)
        v = (s_norm2 / (1.0 + s_norm2)) * (s / s_norm)
        return v

    # Equation (2)
    def affine(self, x):
        # print(self.W.shape,"123")  128*1152*10*16
        x = x.unsqueeze(2).expand(-1, -1, 10, -1).unsqueeze(-1)
        # print(self.W.shape, "self.W")
        # print(x.shape, "1223132")
        x = self.W @ x
        # x = self.W @ x.unsqueeze(2).expand(-1, -1, 10, -1).unsqueeze(-1)
        # print(x.shape, "141414")
        return x.squeeze()

    # Equation (3)
    @staticmethod
    def softmax(x, dim=-1):
        exp = torch.exp(x)
        return exp / torch.sum(exp, dim, keepdim=True)

    # Procedure 1 - Routing algorithm.
    def routing(self, u, r, l):
        b = Variable(torch.zeros(u.size()[0], l[0], l[1]), requires_grad=False).cuda()  # torch.Size([?, 1152, 10])

        for iteration in range(r):
            c = Routing.softmax(b)  # torch.Size([?, 1152, 10])
            # print(c.shape, "ccc")
            s = (c.unsqueeze(-1).expand(-1, -1, -1, u.size()[-1]) * u).sum(1)  # torch.Size([?, 1152, 16])

            v = Routing.squash(s)  # torch.Size([?, 10, 16])
            # print(v.shape, s.shape, "sss")
            b += (u * v.unsqueeze(1).expand(-1, l[0], -1, -1)).sum(-1)
            # print(b.shape, "bbb")
        return v

    def forward(self, x, n_routing_iter):
        x = x.view((-1, self.n_capsules_before, self.caps_dim_before))
        # print(x.shape)
        x = self.affine(x)  # torch.Size([?, 1152, 10, 16])
        # print(x.shape)
        x = self.routing(x, n_routing_iter, (self.n_capsules_before, self.n_capsules_after))
        return x

# Norm
class Norm(torch.nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        x = torch.norm(x, p=2, dim=-1)
        return x

# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, in_features, out_features, output_size=INPUT_SIZE):
        super(Decoder, self).__init__()
        self.decoder = self.assemble_decoder(in_features, out_features)
        self.output_size = output_size

    def assemble_decoder(self, in_features, out_features):
        HIDDEN_LAYER_FEATURES = [512, 1024]
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, HIDDEN_LAYER_FEATURES[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_FEATURES[0], HIDDEN_LAYER_FEATURES[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_FEATURES[1], out_features),
            torch.nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = x[np.arange(0, x.size()[0]), y.cpu().data.numpy(), :].cuda()
        x = self.decoder(x)
        x = x.view(*((-1,) + self.output_size))
        return x

# CapsNet
class CapsNet(torch.nn.Module):
    def __init__(self, input_shape=INPUT_SIZE, n_routing_iter=3, use_reconstruction=True):
        super(CapsNet, self).__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.n_routing_iter = n_routing_iter
        self.use_reconstruction = use_reconstruction

        self.conv1 = Conv1(input_shape[0], 256, 9)
        self.primary_capsules = PrimaryCapsules(
            input_shape=(256, 20, 20),
            capsule_dim=8,
            out_channels=32,
            kernel_size=9,
            stride=2
        )
        self.routing = Routing(
            caps_dim_before=8,
            caps_dim_after=16,
            n_capsules_before=6 * 6 * 32,
            n_capsules_after=10
        )
        self.norm = Norm()

        if (self.use_reconstruction):
            self.decoder = Decoder(16, int(np.prod(input_shape)))

    def n_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])

    def forward(self, x, y=None):
        # print(x.shape)
        conv1 = self.conv1(x)
        # print(conv1.shape)
        primary_capsules = self.primary_capsules(conv1)
        digit_caps = self.routing(primary_capsules, self.n_routing_iter)
        # print(digit_caps.shape, "digit")
        scores = self.norm(digit_caps)
        # print(scores.shape, "ssdsdsdsdsdsdsds")

        if (self.use_reconstruction and y is not None):
            reconstruction = self.decoder(digit_caps, y).view((-1,) + self.input_shape)
            return scores, reconstruction

        return scores

# Margin Loss
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

class MarginLoss(torch.nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lamb=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lamb = lamb

    # Equation (4)
    def forward(self, scores, y):
        y = Variable(to_categorical(y, 10))

        Tc = y.float()
        loss_pos = torch.pow(torch.clamp(self.m_pos - scores, min=0), 2)
        loss_neg = torch.pow(torch.clamp(scores - self.m_neg, min=0), 2)
        loss = Tc * loss_pos + self.lamb * (1 - Tc) * loss_neg
        loss = loss.sum(-1)
        return loss.mean()

# Reconstruction Loss
class SumSquaredDifferencesLoss(torch.nn.Module):
    def __init__(self):
        super(SumSquaredDifferencesLoss, self).__init__()

    def forward(self, x_reconstruction, x):
        loss = torch.pow(x - x_reconstruction, 2).sum(-1).sum(-1)
        return loss.mean()

# Total Loss
class CapsNetLoss(torch.nn.Module):
    def __init__(self, reconstruction_loss_scale=0.0005):
        super(CapsNetLoss, self).__init__()
        self.digit_existance_criterion = MarginLoss()
        self.digit_reconstruction_criterion = SumSquaredDifferencesLoss()
        self.reconstruction_loss_scale = reconstruction_loss_scale

    def forward(self, x, y, x_reconstruction, scores):
        margin_loss = self.digit_existance_criterion(y_pred.cuda(), y)
        reconstruction_loss = self.reconstruction_loss_scale * \
                              self.digit_reconstruction_criterion(x_reconstruction, x)
        loss = margin_loss + reconstruction_loss
        return loss, margin_loss, reconstruction_loss

# Train
model = CapsNet().cuda(0)
# print(model)
# print('Number of Parameters: %d' % model.n_parameters())

criterion = CapsNetLoss()

# Optimizer
def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    if (staircase):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
    else:
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

    return optimizer


LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-08
)


# Training
def save_checkpoint(epoch, train_accuracy, test_accuracy, model, optimizer, path=None):
    if (path is None):
        path = 'checkpoint-%f-%04d.pth' % (test_accuracy, epoch)
    state = {
        'epoch': epoch,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, path)


def show_example(model, x, y, x_reconstruction, y_pred):
    x = x.squeeze().cpu().data.numpy()
    y = y.cpu().data.numpy()
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)
    y_pred = y_pred.cpu().data.numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.show()

def test(model, loader):
    metrics = defaultdict(lambda:list())
    for batch_id, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x = Variable(x).float().cuda()
        y = Variable(y).cuda()
        y_pred, x_reconstruction = model(x, y)
        _, y_pred = torch.max(y_pred, -1)
        metrics['accuracy'].append((y_pred == y).cpu().data.numpy())
    metrics['accuracy'] = np.concatenate(metrics['accuracy']).mean()
    return metrics


global_epoch = 0
global_step = 0
best_tst_accuracy = 0.0
history = defaultdict(lambda:list())
COMPUTE_TRN_METRICS = False

n_epochs = 1500  # Number of epochs not specified in the paper

for epoch in range(n_epochs):
    # print('Epoch %d (%d/%d):' % (global_epoch + 1, epoch + 1, n_epochs))
    # print(next(enumerate(trn_loader)))
    for batch_id, (x, y) in tqdm(enumerate(trn_loader), total=len(trn_loader)):
        optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1,0.90)  # Configurations not specified in the paper
        x = Variable(x).float().cuda()
        y = Variable(y).cuda()

        y_pred, x_reconstruction = model(x, y)
        loss, margin_loss, reconstruction_loss = criterion(x, y, x_reconstruction, y_pred.cuda())

        history['margin_loss'].append(margin_loss.cpu().data.numpy())
        history['reconstruction_loss'].append(reconstruction_loss.cpu().data.numpy())
        history['loss'].append(loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

    trn_metrics = test(model, trn_loader) if COMPUTE_TRN_METRICS else None
    tst_metrics = test(model, tst_loader)

    print('Margin Loss: %f' % history['margin_loss'][-1])
    print('Reconstruction Loss: %f' % history['reconstruction_loss'][-1])
    print('Loss: %f' % history['loss'][-1])
    print('Train Accuracy: %f' % (trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0))
    print('Test Accuracy: %f' % tst_metrics['accuracy'])

    print('Example:')
    idx = np.random.randint(0, len(x))
    show_example(model, x[idx], y[idx], x_reconstruction[idx], y_pred[idx])

    if (tst_metrics['accuracy'] >= best_tst_accuracy):
        best_tst_accuracy = tst_metrics['accuracy']
        save_checkpoint(
            global_epoch + 1,
            trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0,
            tst_metrics['accuracy'],
            model,
            optimizer
        )
    global_epoch += 1

# Loss Curve
def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean

n_points_avg = 10
n_points_plot = 1000
plt.figure(figsize=(20, 10))

curve = np.asarray(history['loss'])[-n_points_plot:]
avg_curve = compute_avg_curve(curve, n_points_avg)
plt.plot(avg_curve, '-g')

curve = np.asarray(history['margin_loss'])[-n_points_plot:]
avg_curve = compute_avg_curve(curve, n_points_avg)
plt.plot(avg_curve, '-b')

curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
avg_curve = compute_avg_curve(curve, n_points_avg)
plt.plot(avg_curve, '-r')

plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
plt.show()
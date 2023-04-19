import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def OP(f, K, labels):
    Q_2 = -f
    Q_2 = F.softmax(Q_2, 1)+1e-18
    labels = labels.long()
    l1 = F.nll_loss(Q_2.log(), labels.long())
    return l1

def OP_W(f, K, labels):
    Q_1 = F.softmax(f, 1)+1e-18
    Q_2 = F.softmax(-f, 1)+1e-18
    w_ = torch.div(1, Q_2)

    w_ = w_ + 1
    w = F.softmax(w_,1)

    w = torch.mul(Q_1,w)+1e-6
    w_1 = torch.mul(w, Q_2.log())
    l2 = F.nll_loss(w_1, labels.long())
    return l2

def chosen_loss_c(f, K, labels, method):
    final_loss = 0
    if method == 'OP':
        final_loss = OP(f, K, labels)
    elif method == 'OP-W':
        final_loss = OP_W(f, K, labels)
    return final_loss


def accuracy_check(loader, model):
    sm = F.softmax
    total, num_samples = 0, 0
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images)
        sm_outputs = sm(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples



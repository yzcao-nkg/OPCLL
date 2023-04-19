from utils_data import *
from utils_algo import *
from models import *
import argparse, time, os

parser = argparse.ArgumentParser(
	prog='order-preservingcomplementary-label learning demo file.',
	usage='Demo with complementary labels.',
	description='A simple demo file with Kuzushiji-MNIST dataset.',
	epilog='end',
	add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=5e-5, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=256, type=int)
parser.add_argument('-me', '--method',  default='OP', choices=['OP','OP-W'], type=str, required=False)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=150)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)

device = "cpu"
args = parser.parse_args()
np.random.seed(10); torch.manual_seed(10); torch.cuda.manual_seed_all(10)


full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_kmnist_data(
    batch_size=args.batch_size)

ordinary_train_loader, complementary_train_loader, ccp, = prepare_train_loaders(
    full_train_loader=full_train_loader, batch_size=args.batch_size,
    ordinary_train_dataset=ordinary_train_dataset)

model = mlp_model(input_dim=28 * 28, hidden_dim=500, output_dim=K)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
test_accuracy = accuracy_check(loader=test_loader, model=model)
print('Epoch: 0. Te Acc: {}'.format(test_accuracy))

for epoch in range(args.epochs):
    train_loss = 0
    for i, (images, labels) in enumerate(complementary_train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = chosen_loss_c(f=outputs, K=K, labels=labels, method=args.method)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    train_accuracy = accuracy_check(loader=train_loader, model=model)
    test_accuracy = accuracy_check(loader=test_loader, model=model)
    print('Epoch: {}. Te Acc: {}.'.format(epoch + 1, test_accuracy))

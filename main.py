import INN

batch_size = 100
epoch = 10
trainset = INN.datasets.MNIST(train=True)
testset = INN.datasets.MNIST(train=False)

train_loader = INN.dataloaders.DataLoader(trainset, batch_size, shuffle=True)
test_loader = INN.dataloaders.DataLoader(testset, batch_size, shuffle=False)

model = INN.models.MLP((1000, 1000, 10), activation=INN.functions.relu)
optimizer = INN.optimizers.Adam().setup(model)

for i in range(epoch):
    sum_loss, sum_acc = 0, 0
    sum_loss1, sum_acc1 = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = INN.functions.softmax_cross_entropy(y, t)
        acc = INN.functions.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += loss
        sum_acc += acc
    print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
    sum_loss, sum_acc = 0, 0
    sum_loss1, sum_acc1 = 0, 0
    with INN.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = INN.functions.softmax_cross_entropy(y, t)
            acc = INN.functions.accuracy(y, t)
            sum_loss += loss
            sum_acc += acc
    print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')
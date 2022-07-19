import INN

batch_size = 100
epoch = 10
trainset = INN.datasets.MNIST(train=True)
testset = INN.datasets.MNIST(train=False)

train = INN.dataloaders.DataLoader(trainset, batch_size, shuffle=True)
test = INN.dataloaders.DataLoader(testset, batch_size, shuffle=False)

model = INN.models.MLP((1000, 1000, 10), activation=INN.functions.relu)
optimizer = INN.optimizers.Adam().setup(model)

for i in range(epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train:
        y = model(x)
        loss = INN.functions.softmax_cross_entropy(y, t)
        acc = INN.functions.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += loss * batch_size
        sum_acc += acc * batch_size
    print(f'train loss {sum_loss / len(trainset)} accuracy {sum_acc / len(trainset)}')
    sum_loss, sum_acc = 0, 0

    with INN.no_grad():
        for x, t in test:
            y = model(x)
            loss = INN.functions.softmax_cross_entropy(y, t)
            acc = INN.functions.accuracy(y, t)
            sum_loss += loss * batch_size
            sum_acc += acc * batch_size
    print(f'test loss {sum_loss / len(testset)} accuracy {sum_acc / len(testset)}')
import lijnn

batch_size = 100
epoch = 10
trainset = lijnn.datasets.MNIST(train=True)
testset = lijnn.datasets.MNIST(train=False)

train_loader = lijnn.iterators.iterator(trainset, batch_size, shuffle=True)
test_loader = lijnn.iterators.iterator(testset, batch_size, shuffle=False)

model = lijnn.models.MLP((1000, 1000, 10), activation=lijnn.functions.relu)
optimizer = lijnn.optimizers.Adam().setup(model)

if lijnn.cuda.gpu_enable:
    model.to_gpu()
    train_loader.to_gpu()
    test_loader.to_gpu()

for i in range(epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = lijnn.functions.softmax_cross_entropy(y, t)
        acc = lijnn.functions.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += loss.data
        sum_acc += acc.data
    print(f"epoch {i + 1}")
    print(f'train loss {sum_loss / train_loader.max_iter} accuracy {sum_acc / train_loader.max_iter}')
    sum_loss, sum_acc = 0, 0

    with lijnn.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = lijnn.functions.softmax_cross_entropy(y, t)
            acc = lijnn.functions.accuracy(y, t)
            sum_loss += loss.data
            sum_acc += acc.data
    print(f'test loss {sum_loss / test_loader.max_iter} accuracy {sum_acc / test_loader.max_iter}')
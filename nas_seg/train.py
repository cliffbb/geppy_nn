



def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    # criterion = nn.NLLLoss2d(weight=weights)
    criterion = nn.CrossEntropyLoss()
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            # loss = CrossEntropy2d(output, target, weight=weights)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                # clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                # rgb = np.asarray(np.transpose(data.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                        e, epochs, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))

            #                 plt.plot(mean_losses[:iter_]) and plt.show()
            #                 fig = plt.figure()
            #                 fig.add_subplot(131)
            #                 plt.imshow(rgb)
            #                 plt.title('RGB')
            #                 fig.add_subplot(132)
            #                 plt.imshow(convert_to_color(gt))
            #                 plt.title('Ground truth')
            #                 fig.add_subplot(133)
            #                 plt.title('Prediction')
            #                 plt.imshow(convert_to_color(pred))
            #                 plt.show()
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
            torch.save(net.state_dict(), './segnet256_epoch{}_{}'.format(e, acc))
    torch.save(net.state_dict(), './segnet_final')
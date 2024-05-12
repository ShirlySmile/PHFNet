import torch.nn as nn
from utils import *
from datatools import *
import random
import os, cv2, torch, time, logging, datetime, argparse, math
import torch_optimizer as optim
from adabelief_pytorch import AdaBelief


# MUUFL
ColorBoard = np.array(
    [[0, 0, 0], [0, 127, 0], [0, 255, 0], [0, 255, 255], [255, 204, 0], [255, 0, 50], [0, 0, 205], [102, 0, 205], [255, 127, 153],  [203, 102, 0],
     [255, 255, 0], [204, 26, 10]]
)

# # 2013
# ColorBoard = np.array(
#         [[0, 0, 0], [76, 176, 58], [124, 192, 46], [63, 133, 88], [57, 133, 61], [139, 70, 46], [112, 195, 199], [255, 255, 255], [201, 169, 206],  [219, 39, 40],
# [115, 32, 35], [59, 94, 171], [217, 225, 57], [215, 131, 37], [76, 43, 126], [228, 97, 77]])

#

# Augsburg
# ColorBoard = np.array([[0, 0, 0], [41, 101, 58], [238, 44, 42], [235, 233, 79], [140, 198, 65], [151, 84, 55], [112, 190, 68], [123, 207, 216], [48, 121, 188]])

ColorBoard = ColorBoard[:,(2,1,0)] #RGB转化成BGR


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='PHFNet', help='PHFNet   neural network used in training')

    parser.add_argument('--dataname', type=str, default='MUUFL', help='dataset used for training: Berlin/Augsburg/MUUFL/2018')

    parser.add_argument('--dataset_mode', type=str, default='sample',
                        help='the data partitioning strategy: sample/fixed   sample/fixed/')
    parser.add_argument('--partition', type=str, default='Cls_Num_Same', help='the data partitioning strategy: Cls_Num_Same/Cls_Rate_Same')
    parser.add_argument('--Traindata_Num', type=float, default=20, help='the split dataset per-class number')
    parser.add_argument('--Traindata_Rate', type=float, default=0.02, help='the split dataset rate')

    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='input batch size for training (default: 64)')

    parser.add_argument('--LR', type=float, default=0.01, help='learning rate (default: 0.1) Berlin:0.0005/45')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')


    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

    # 保存模型信息0.001
    parser.add_argument('--logdir', type=str, default='./log/', help='save message')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    args = get_args()
    device = torch.device(args.device)
    tag = str(datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
    logdir = args.logdir + '/' + args.dataname + '/' + tag + '-' +  args.model + '/'
    setlog(device, logdir)

    print("**Basic Setting...")
    logger.info("**Basic Setting...")
    print('  ', args)
    logging.info(args)

    seed = 1234


    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    start_time = datetime.datetime.now().strftime('%F %T')
    print("程序开始运行时间：" + start_time)
    logger.info("程序开始运行时间：" + start_time)


    if args.dataset_mode == "fixed":
        img1, img2, labels_train, labels_test = getdata_fixed(args.dataname, 0.005, args.patch_size, args.mode)
        labels = labels_train + labels_test
    elif args.dataset_mode == "sample":
        img1, img2, labels = getdata_sample(args.dataname, 0.005, args.patch_size, args.mode)
        labels_train, labels_test = splitmaskdata_sample(labels, args.Traindata_Rate, args.Traindata_Num, args.partition, seed)

    cls_num = len(np.unique(labels_train))-1
    train_loader = getmask_training_dataloader(img1, img2, labels_train, seed, args.BATCH_SIZE, samplerstate=False)
    test_loader = getmask_test_dataloader(img1, img2, labels_test)
    all_loader = getmask_all_dataloader(img1, img2, labels_test)

    labels_t = torch.from_numpy(np.copy(labels_train)).type(torch.LongTensor)

    net, criterion_extra = getnet(args.model, img1.shape[0], img2.shape[0], cls_num, args.patch_size)


    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.LR, weight_decay=0.0005)

    trainstart = time.time()
    print("训练开始时间：" + start_time)
    logger.info("训练开始时间：" + start_time)


    for epoch in range(args.epochs):
        valid_batch = iter(test_loader)  # 验证集迭代器
        for step, (img1_patch, img2_patch, label) in enumerate(train_loader, start=epoch * len(
                train_loader)):

            zero = torch.zeros_like(label)
            ones = torch.ones_like(label)
            output = net(img1_patch, img2_patch)
            if criterion_extra:
                loss = criterion_extra(output, label)
            else:
                loss = criterion(output, label)

            if isinstance(output, tuple):
                output = output[0]

            train_pred_y = torch.max(output, 1)[1].cuda().data

            num = float(len(torch.where(label != -1)[1]))

            mask_pred = torch.where(label != -1, ones, zero)
            train_acc = ((train_pred_y.squeeze() == label.squeeze()) * mask_pred).sum() / num

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 2 == 0:
                net.eval()
                img1_test, img2_test, label_t = next(valid_batch)
                with torch.no_grad():
                    # test_output = net(img1_test)  # net output
                    test_output = net(img1_test, img2_test)
                if isinstance(test_output, tuple):  # For multiple outputs
                    test_output = test_output[0]
                pred_y = torch.max(test_output, 1)[1].cuda().data

                num = float(len(torch.where(label_t != -1)[1]))

                zero = torch.zeros_like(label_t)
                ones = torch.ones_like(label_t)

                mask_pred = torch.where(label_t != -1, ones, zero)
                accuracy = ((pred_y.squeeze() == label_t.squeeze()) * mask_pred).sum() / num

                print(
                    '[Epoch:%3d] || step: %3d || LR: %.6f  || train loss: %.4f || train acc: %.4f || test acc: %.4f' % (
                    epoch, step, optimizer.param_groups[0]['lr'], loss.item(), train_acc, accuracy))
                logger.info(
                    '[Epoch:%3d] || step: %3d || LR: %.6f  || train loss: %.4f || train acc: %.4f || test acc: %.4f' % (
                    epoch, step, optimizer.param_groups[0]['lr'], loss.item(), train_acc, accuracy))
                net.train()
            adjust_learning_rate(args, optimizer, train_loader, step)

    trainend = time.time()

    # # 保存模型
    savedir = logdir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    save_model_path = savedir + 'checkpoint.pth.tar'
    torch.save({'state_dict': net.state_dict()}, save_model_path)

    net2, _ = getnet(args.model, img1.shape[0], img2.shape[0], cls_num, args.patch_size)
    checkpoint = torch.load(save_model_path)
    net2.load_state_dict(checkpoint['state_dict'])
    OA, AA, Kappa = eval_maskmodel(net2, test_loader, trainstart, trainend, 'test', args.model)
    print('OA:', OA)
    logger.info('OA:%.7f', OA)
    print('AA:', AA)
    logger.info('AA:%.7f', AA)
    print('Kappa:', Kappa)
    logger.info('Kappa:%.7f', Kappa)


    paintmask_predictlabel(all_loader, net2, ColorBoard, labels, savedir)

    print('着色图生成完毕')
    logger.info('着色图生成完毕')
    end_time = datetime.datetime.now().strftime('%F %T')
    print("程序结束运行时间：" + end_time)
    logger.info("程序结束运行时间：" + end_time)

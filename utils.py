import os, cv2, torch, time, logging, math
from sklearn.metrics import confusion_matrix
import numpy as np
from models import PHFNet


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def divisible_pad(image_list, size_divisor=128, to_tensor=True):
    """

    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    if to_tensor:
        storage = torch.FloatStorage._new_shared(len(image_list) * np.prod(max_shape))
        out = torch.Tensor(storage).view([len(image_list), max_shape[0], max_shape[1], max_shape[2]])
        out = out.zero_()
    else:
        out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float64)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def getnet(model, in_channels1, in_channels2, cls_num, patch_size):
    loss = None

    if model == 'PHFNet':
        net = PHFNet(in_channels1, in_channels2, cls_num)

    return net, loss




def setlog(device, logdir):
    mkdirs(logdir)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logdir, 'info.log'),
        format='[%(levelname)s](%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d/ %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.info(device)



def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.BATCH_SIZE / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * args.LR


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def eval_maskmodel(net, test_loader, trainstart, trainend, mode, model, w0=0, w1=0, w2=0):
    """对训练好的模型进行评估"""
    net.cuda()
    teststart = time.time()

    l = 0
    y_pred = []
    net.eval()


    for step, (ms, pan, label) in enumerate(test_loader):
        with torch.no_grad():
            output = net(ms.cuda(), pan.cuda())
            # output = net(ms.cuda())

        if isinstance(output, tuple):  # For multiple outputs
            output = output[0]

        pred_y = torch.max(output, 1)[1].cuda().data

        zero = torch.zeros_like(label)
        ones = torch.ones_like(label)
        w = torch.where(label != -1, ones, zero)

        w = w.byte()
        label = torch.masked_select(label.view(-1), w.view(-1))
        pred_y = torch.masked_select(pred_y.view(-1), w.view(-1))


    # save = softmax(saveoutput)
    if mode == 'test':
        testend = time.time()
        traintime = trainend - trainstart
        testtime = testend - teststart
        print("train time: %.2f S || test time: %.2f S" % (traintime, testtime))
        logger.info("train time: %.2f S || test time: %.2f S" % (traintime, testtime))

    label = label.cpu().numpy()
    pred_y = pred_y.cpu().numpy()
    # ss = np.concatenate((showlabel[:, np.newaxis], y_pred[:, np.newaxis]), axis=1)
    con_mat = confusion_matrix(y_true=label, y_pred=pred_y)
    if mode == 'test':
        print('con_mat', con_mat)
        logger.info(con_mat)
    # 计算性能参数
    all_acr = 0
    p = 0
    column = np.sum(con_mat, axis=0)  # 列求和
    line = np.sum(con_mat, axis=1)  # 行求和
    for i, clas in enumerate(con_mat):
        precise = clas[i]
        all_acr = precise + all_acr

        acr = precise / column[i]
        recall = precise / line[i]

        f1 = 2 * acr * recall / (acr + recall)
        temp = column[i] * line[i]
        p = p + temp
        if mode is 'test':
            print("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
            logger.info("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
    OA = np.trace(con_mat) / np.sum(con_mat)
    AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))  # axis=1 每行求和
    Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
    Kappa = (OA - Pc) / (1 - Pc)


    return OA, AA, Kappa





def paintmask_predictlabel(all_data_loader, net, color_board, labels, savedir):
    """生成着色图"""
    out_color = np.zeros((labels.shape[0], labels.shape[1], 3))
    out_labeledcolor = np.zeros((labels.shape[0], labels.shape[1], 3))

    net.cuda()
    net.eval()


    for step, (img1_patch, img2_patch, gt_xy) in enumerate(all_data_loader):
        with torch.no_grad():
            output = net(img1_patch, img2_patch)
        if isinstance(output, tuple):  # For multiple outputs
            output = output[0]
        pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()


    for i,j in np.argwhere(labels != 0):
        cls = pred_y_numpy[i][j] + 1
        out_labeledcolor[i][j] = color_board[cls]

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            cls = pred_y_numpy[i][j] + 1
            out_color[i][j] = color_board[cls]

    savelabeled_predict_img_path = savedir + 'showlabeled_result.png'
    save_predict_img_path = savedir + 'show_result.png'
    cv2.imwrite(savelabeled_predict_img_path, out_labeledcolor)
    cv2.imwrite(save_predict_img_path, out_color)





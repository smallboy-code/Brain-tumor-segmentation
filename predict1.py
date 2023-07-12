import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
import SimpleITK as sitk

from medpy.metric import binary

def softmax_spec_score(output, target):
    ret = []

    # whole
    o = output > 0;t = target > 0  # ce
    ret += Sens(~o, ~t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += Sens(~o, ~t),
    # active
    o = (output == 3);t = (target == 4)
    ret += Sens(~o, ~t),

    return ret

def hausdorff_distance(lT,lP):
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    return hausdorffcomputer.GetAverageHausdorffDistance()#hausdorffcomputer.GetHausdorffDistance()

def softmax_hd95_score(output, target):
    ret = []

    # whole
    o = output > 0;t = target > 0  # ce
    if o.sum() == 0 or t.sum() == 0:
        ret += 0,
    else:
        ret += binary.hd95(o, t, voxelspacing=None),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    if o.sum() == 0 or  t.sum() == 0:
        ret += 0,
    else:
        ret += binary.hd95(o, t, voxelspacing=None),
    # active
    o = (output == 3);t = (target == 4)
    if o.sum() == 0 or t.sum() == 0:
        ret += 0,
    else:
        ret += binary.hd95(o, t, voxelspacing=None),

    return ret

def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()


def tailor_and_concat(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()

    for i in range(len(temp)):
        temp[i] = model(temp[i])

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]


def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den


def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den

def Sens(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = t.sum() + eps
    return num/den

def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==4)))
    return mIOU_score


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += dice_score(o, t),

    return ret

def softmax_sens_score(output, target):
    ret = []

    # whole
    o = output > 0;t = target > 0  # ce
    ret += Sens(o, t),
    # core
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += Sens(o, t),
    # active
    o = (output == 3);t = (target == 4)
    ret += Sens(o, t),

    return ret

keys = 'whole', 'core', 'enhancing', 'loss'


def validate_softmax(
        valid_loader,
        model,
        load_file,
        multimodel,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly!
        verbose=False,
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=True,  # if you are valid when train
        ):

    H, W, T = 240, 240, 155
    model.eval()
    dice_WT = 0.0
    dice_TC = 0.0
    dice_ET = 0.0
    sens_WT = 0.0
    sens_TC = 0.0
    sens_ET = 0.0
    spec_WT = 0.0
    spec_TC = 0.0
    spec_ET = 0.0
    hd_WT = 0.0
    hd_TC = 0.0
    hd_ET = 0.0
    runtimes = []
    ET_voxels_pred_list = []
    path = "/media/dmia/data11/hh/zdy_new/aixiaotuan/aixiaotuan_seg1.nii.gz"
    mask = sitk.ReadImage(path, sitk.sitkUInt8)
    out_csv = open("output.csv", "w")
    out_csv.writelines("Label,Dice_ET,Dice_WT,Dice_TC,sens_ET,sens_WT,sens_TC,spec_ET,spec_WT,spec_TC,hd_ET,hd_WT,hd_TC"+'\n')
    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        if valid_in_train:
            data = [t.cuda(non_blocking=True) for t in data]
            x, target = data[:2]
        else:
            x = data
            x.cuda()

        if not use_TTA:
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            logit = tailor_and_concat(x, model)

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time/60))
            runtimes.append(elapsed_time)


            if multimodel:
                logit = F.softmax(logit, dim=1)
                output = logit / 4.0

                load_file1 = load_file.replace('7998', '7996')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7997')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
                load_file1 = load_file.replace('7998', '7999')
                if os.path.isfile(load_file1):
                    checkpoint = torch.load(load_file1)
                    model.load_state_dict(checkpoint['state_dict'])
                    print('Successfully load checkpoint {}'.format(load_file1))
                    logit = tailor_and_concat(x, model)
                    logit = F.softmax(logit, dim=1)
                    output += logit / 4.0
            else:
                output = F.softmax(logit, dim=1)


        else:
            x = x[..., :155]
            logit = F.softmax(tailor_and_concat(x, model), 1)  # no flip
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2,)), model).flip(dims=(2,)), 1)  # flip H
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3,)), model).flip(dims=(3,)), 1)  # flip W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(4,)), model).flip(dims=(4,)), 1)  # flip D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3)), model).flip(dims=(2, 3)), 1)  # flip H, W
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 4)), model).flip(dims=(2, 4)), 1)  # flip H, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(3, 4)), model).flip(dims=(3, 4)), 1)  # flip W, D
            logit += F.softmax(tailor_and_concat(x.flip(dims=(2, 3, 4)), model).flip(dims=(2, 3, 4)), 1)  # flip H, W, D
            output = logit / 8.0  # mean

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)
        #print(output.shape, target.cpu().detach().numpy().squeeze().shape)
        if np.count_nonzero(output == 3) <= 500:
            output[output == 3] = 1
        if np.count_nonzero(output == 1) + np.count_nonzero(output == 3) <= 1000:
            output[output == 1] = 2
            output[output == 3] = 2
        dice_ret = softmax_output_dice(output, target.cpu().detach().numpy().squeeze())
        sens_ret = softmax_sens_score(output, target.cpu().detach().numpy().squeeze())
        spec_ret = softmax_spec_score(output, target.cpu().detach().numpy().squeeze())
        hd_ret = softmax_hd95_score(output, target.cpu().detach().numpy().squeeze())
        dice_WT += dice_ret[0]
        dice_TC += dice_ret[1]
        dice_ET += dice_ret[2]
        sens_WT += sens_ret[0]
        sens_TC += sens_ret[1]
        sens_ET += sens_ret[2]
        spec_WT += spec_ret[0]
        spec_TC += spec_ret[1]
        spec_ET += spec_ret[2]
        hd_WT += hd_ret[0]
        hd_TC += hd_ret[1]
        hd_ET += hd_ret[2]
        name = str(i)
        out_csv.writelines(names[i] + "," + str(dice_ret[2]) + "," + str(dice_ret[0]) + "," + str(dice_ret[1]) + "," + str(sens_ret[2]) + "," + str(sens_ret[0]) + "," + \
                str(sens_ret[1]) + "," + str(spec_ret[2]) + "," + str(spec_ret[0]) + "," + str(spec_ret[1]) + "," + str(hd_ret[2]) + "," + str(hd_ret[0]) + "," + str(hd_ret[1]) + "\n")
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print( msg, dice_ret, sens_ret, spec_ret, hd_ret)


        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name + '.nii.gz')
                output = output.astype('float')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

                seg_img[output == 1] = 1
                seg_img[output == 2] = 2
                seg_img[output == 3] = 4
                if np.count_nonzero(output == 3) <= 350:
                    seg_img[output == 3] = 1
                seg_img = seg_img.transpose(2, 1, 0)
                ys_pd_itk = sitk.GetImageFromArray(seg_img)
                ys_pd_itk.SetSpacing(mask.GetSpacing())
                ys_pd_itk.SetOrigin(mask.GetOrigin())
                ys_pd_itk.SetDirection(mask.GetDirection())

                if verbose:
                    print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
                    print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
                          np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
                # nib.save(nib.Nifti1Image(seg_img, None), oname)
                sitk.WriteImage(ys_pd_itk, oname)
                print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255
                    Snapshot_img = Snapshot_img.transpose(1, 0, 2, 3)
                    for frame in range(T):
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
    out_csv.writelines("mean" + "," + str(dice_ET/1251.0) + "," + str(dice_WT/1251.0) + "," + str(dice_TC/1251.0) + "," + str(sens_ET/1251.0) + "," + str(sens_WT/1251.0) + "," + \
            str(sens_TC/1251.0) + "," + str(spec_ET/1251.0) + "," + str(spec_WT/1251.0) + "," + str(spec_TC/1251.0) + "," + str(hd_ET/1251.0) + "," + str(hd_WT/1251.0) + "," + str(hd_TC/1251.0))
    #print("avg_WT:{0:.4f};avg_TC:{0:.4f};avg_ET:{0:.4f}".format(WT/100.0, TC/100.0, ET/100.0))
    out_csv.close()
    print('runtimes:', sum(runtimes)/len(runtimes))

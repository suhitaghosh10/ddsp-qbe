import os
import time
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call

import torch
from torch.utils.tensorboard import SummaryWriter

from fusion_synthesis.utility.checkpoint import Checkpoint
from fusion_synthesis.utility import utils

def train(args, model, loss_func, loader_train, loader_test, generate_files=False):
    # saver
    saver = Checkpoint(args)
    train_writer = SummaryWriter(log_dir=os.path.join(saver.expdir, "train"))
    val_writer = SummaryWriter(log_dir=os.path.join(saver.expdir, "val"))

    # model size
    print('model parameters:', utils.get_network_params({'model': model}))

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, eps=1e-02,
    #                                                        verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, eps=1e-02, verbose=True)
    # best_loss = np.inf
    best_hnr = -np.inf
    num_batches = len(loader_train)
    print('batches-', num_batches)
    use_emo = args.loss.use_emo_loss
    model.train()

    for epoch in range(args.train.epochs):
        print('======= epoch {}/{} ======='.format(epoch, args.train.epochs))
        loss_epch = 0.
        loss_mss_epch = 0.
        loss_f0_epch = 0.
        loss_krtss_epch = 0.
        loss_jttr_epch = 0.
        loss_shmmr_epch = 0.
        loss_emo = 0.
        ctr_stime = time.time()
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad(set_to_none=True)

            # unpack data
            for key in data.keys():
                if key != 'name':
                    data[key] = data[key].to(args.device).float()

            if use_emo:
                signal, f0_pred, _, _, emo_rep = model(x6=data['w6'], x12=data['w12'], x6_emo1=data['e6_1'],
                                                       x6_emo2=data['e6_2'], f0_norm=data['norm_f0'])
                # loss
                loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer, prosody_leakage_loss) = loss_func(
                    signal, data['audio'], f0_pred, data['f0'], emo_rep=emo_rep, is_val=False)
            else:
                signal, f0_pred, _, _, _ = model(x6=data['w6'], x12=data['w12'], f0_norm=data['norm_f0'])
                # loss
                loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer, prosody_leakage_loss) = loss_func(
                    signal, data['audio'], f0_pred, data['f0'], emo_rep=None, is_val=False)




            loss_epch += loss
            loss_mss_epch += loss_mss
            loss_f0_epch += loss_f0
            loss_krtss_epch += loss_kurtosis
            loss_jttr_epch += loss_jitter
            loss_shmmr_epch += loss_shimmer
            loss_emo += prosody_leakage_loss

            # handle nan loss
            if torch.isnan(loss):
                print(signal, f0_pred)
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                loss.backward()
                optimizer.step()

            if saver.global_step % args.train.interval_log == 0:
                ctr_etime = time.time()
                time_taken = ctr_etime - ctr_stime

                print(
                    'Batch {}/{} | {} | train loss: {:.6f}, mss: {:.6f}, F0: {:.6f}, jitter: {:.6f}, shimmer: {:.6f} loss_emo {:.6f}| iteration: {} | avg. time: {:.2f}s'.format(
                        batch_idx,
                        num_batches,
                        saver.expdir,
                        loss.item(),
                        loss_mss.item(),
                        loss_f0.item(),
                        loss_jitter.item(),
                        loss_shimmer.item(),
                        loss_emo.item(),
                        saver.global_step,
                        time_taken/args.train.interval_log
                    )
                )
                ctr_stime = time.time()  # when epoch doesnt end

        test_loss, test_loss_mss, test_loss_krtss, test_loss_f0, test_loss_jttr, test_loss_shmmr, test_loss_emo, test_hnr, time_taken = validation(
            args, model, loss_func, loader_test,
            out_path=os.path.join(args.env.expdir, str(epoch)),
            generate_flag=generate_files,
            epoch=epoch)
        print(
            'epoch {}/{} val loss: {:.3f}, mss loss: {:.3f}, F0 loss: {:.3f}, HNR: {:.3f} | time: {:.2f}'.format(
                epoch,
                args.train.epochs,
                loss.item(),
                loss_mss.item(),
                loss_f0.item(),
                test_hnr,
                time_taken
            )
        )
        test_loss_to_save = test_loss_mss + test_loss_f0

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(metrics=test_hnr)
        after_lr = optimizer.param_groups[0]["lr"]
        print("AdamW lr %.4f -> %.4f" % (before_lr, after_lr))

        # log in tensorboard
        train_writer.add_scalar("Loss", loss_epch/num_batches, epoch)
        train_writer.add_scalar("MSS Loss", loss_mss_epch/num_batches, epoch)
        train_writer.add_scalar("F0 Loss", loss_f0_epch/num_batches, epoch)
        train_writer.add_scalar("Kurtosis Loss", loss_krtss_epch/num_batches, epoch)
        train_writer.add_scalar("Jitter Loss", loss_jttr_epch/num_batches, epoch)
        train_writer.add_scalar("Shimmer Loss", loss_shmmr_epch/num_batches, epoch)
        train_writer.add_scalar("Emo Loss", loss_emo / num_batches, epoch)
        train_writer.add_scalar("LR", before_lr, epoch)
        train_writer.flush()

        val_writer.add_scalar("Loss", test_loss, epoch)
        val_writer.add_scalar("MSS Loss", test_loss_mss, epoch)
        val_writer.add_scalar("F0 Loss", test_loss_f0, epoch)
        val_writer.add_scalar("Kurtosis Loss", test_loss_krtss, epoch)
        val_writer.add_scalar("Jitter Loss", test_loss_jttr, epoch)
        val_writer.add_scalar("Shimmer Loss", test_loss_shmmr, epoch)
        val_writer.add_scalar("Emo Loss", test_loss_emo, epoch)
        val_writer.add_scalar("HNR", test_hnr / num_batches, epoch)

        val_writer.flush()

        #change to training mode
        model.train()

        # save best model
        # if test_loss_to_save < best_loss:
        #     saver.save_models(
        #         {'vocoder': model}, postfix='best')
        #     best_loss = test_loss_to_save
        if test_hnr > best_hnr:
            saver.save_models(
                {'vocoder': model}, postfix='best')
            best_hnr = test_hnr

    train_writer.close()
    val_writer.close()


def validation(args, model, loss_func, loader_test, epoch, out_path, generate_flag=False):
    model.eval()

    # losses
    test_loss = 0.
    test_loss_mss = 0.
    test_loss_f0 = 0.
    test_loss_krtss = 0.
    test_loss_jttr = 0.
    test_loss_shmmr = 0.
    test_loss_emo = 0.
    test_hnr = 0.

    num_batches = len(loader_test)

    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()

            st_time = time.time()
            if args.loss.use_emo_loss:
                signal, f0_pred, _, _, emo_rep = model(x6=data['w6'], x12=data['w12'], x6_emo1=data['e6_1'],
                                                       x6_emo2=data['e6_2'], f0_norm=data['norm_f0'])
                loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer, prosody_leakage_loss) = loss_func(
                    signal, data['audio'], f0_pred, data['f0'], emo_rep=emo_rep, is_val=True)
            else:
                signal, f0_pred, _, _, _ = model(x6=data['w6'], x12=data['w12'], f0_norm=data['norm_f0'])
                loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer, prosody_leakage_loss) = loss_func(
                    signal, data['audio'], f0_pred, data['f0'], emo_rep=None, is_val=True)

            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal = signal[:, :min_len]
            data['audio'] = data['audio'][:, :min_len]

            hnr = get_signals_hnr(signal.cpu().numpy())

            test_loss += loss.item()
            test_loss_mss += loss_mss.item()
            test_loss_f0 += loss_f0.item()
            test_loss_krtss += loss_kurtosis.item()
            test_loss_jttr += loss_jitter.item()
            test_loss_shmmr += loss_shimmer.item()
            test_loss_emo += prosody_leakage_loss.item()
            test_hnr += hnr

            if generate_flag:
                print(' [*] output folder:', args.ex)
                os.makedirs(out_path, exist_ok=True)
                # path
                path_pred = os.path.join(out_path, fn + '.wav')
                pred = utils.convert_tensor_to_numpy(signal)
                # save
                sf.write(path_pred, pred, args.data.sampling_rate)
                if epoch == 0:
                    path_anno = os.path.join(out_path, 'anno', fn + '.wav')
                    os.makedirs(os.path.dirname(path_anno), exist_ok=True)
                    anno = utils.convert_tensor_to_numpy(data['audio'])
                    sf.write(path_anno, anno, args.data.sampling_rate)

    test_loss /= num_batches
    test_loss_mss /= num_batches
    test_loss_f0 /= num_batches
    test_loss_krtss /= num_batches
    test_loss_jttr /= num_batches
    test_loss_shmmr /= num_batches
    test_loss_emo /= num_batches
    test_hnr /= num_batches

    return test_loss, test_loss_mss, test_loss_krtss, test_loss_f0, test_loss_jttr, test_loss_shmmr, test_loss_emo, test_hnr, (ed_time - st_time)

def get_signals_hnr(signal, sr=16000):
    len_sig = signal.shape[0]
    hnr = [get_hnr(signal[i], sr) for i in range(len_sig)]
    hnr = sum(hnr) / len(hnr)
    return hnr

def get_hnr(wav, sr=16000):
    sound = parselmouth.Sound(wav, sr)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 80, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return hnr

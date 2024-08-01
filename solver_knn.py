import os
import time
import numpy as np
import soundfile as sf

from fusion_synthesis.logger.saver import utils, Saver
import torch
from torch.utils.tensorboard import SummaryWriter

def infer_on_validation(args, model, loss_func, loader_test, epoch, path_gendir='gen', is_part=False, generate_flag=False):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    test_loss_mss = 0.
    test_loss_f0 = 0.
    test_loss_krtss = 0.
    test_loss_jttr = 0.
    test_loss_shmmr = 0.

    # intialization
    num_batches = len(loader_test)

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()

            # forward
            st_time = time.time()
            signal, f0_pred, _, (s_h, s_n) = model(mel=data['mel'], mel12=data['mel12'], f0_norm=data['norm_f0'])
            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal = signal[:, :min_len]
            data['audio'] = data['audio'][:, :min_len]

            # loss
            # loss, (loss_mss, loss_f0, loss_jitter, loss_silence, loss_shimmer) = loss_func(
            loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer) = loss_func(
                signal, data['audio'], f0_pred, data['f0'], is_val=True)

            test_loss += loss.item()
            test_loss_mss += loss_mss.item()
            test_loss_f0 += loss_f0.item()
            test_loss_krtss += loss_kurtosis.item()
            test_loss_jttr += loss_jitter.item()
            test_loss_shmmr += loss_shimmer.item()

            if generate_flag:
                print(' [*] output folder:', args.ex)
                os.makedirs(path_gendir, exist_ok=True)
                # path
                path_pred = os.path.join(path_gendir, fn + '.wav')
                if is_part:
                    path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
                    path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')

                os.makedirs(os.path.dirname(path_pred), exist_ok=True)
                if is_part:
                    os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

                # to numpy
                pred = utils.convert_tensor_to_numpy(signal)
                if is_part:
                    pred_n = utils.convert_tensor_to_numpy(s_n)
                    pred_h = utils.convert_tensor_to_numpy(s_h)

                # save
                sf.write(path_pred, pred, args.data.sampling_rate)
                if epoch == 0:
                    path_anno = os.path.join(path_gendir, 'anno', fn + '.wav')
                    os.makedirs(os.path.dirname(path_anno), exist_ok=True)
                    anno = utils.convert_tensor_to_numpy(data['audio'])
                    sf.write(path_anno, anno, args.data.sampling_rate)
                if is_part:
                    sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                    sf.write(path_pred_h, pred_h, args.data.sampling_rate)

    # report
    test_loss /= num_batches
    test_loss_mss /= num_batches
    test_loss_f0 /= num_batches
    test_loss_krtss /= num_batches
    test_loss_jttr /= num_batches
    test_loss_shmmr /= num_batches

    return test_loss, test_loss_mss, test_loss_krtss, test_loss_f0, test_loss_jttr, test_loss_shmmr, (ed_time - st_time)


def train(args, model, loss_func, loader_train, loader_test, is_part=False, generate_files=False):
    # saver
    saver = Saver(args)
    train_writer = SummaryWriter(log_dir=os.path.join(saver.expdir, "train"))
    val_writer = SummaryWriter(log_dir=os.path.join(saver.expdir, "val"))

    # model size
    print('model parameters:', utils.get_network_paras_amount({'model': model}))

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, eps=1e-02,
                                                           verbose=True)

    best_loss = np.inf
    num_batches = len(loader_train)
    print('batches-', num_batches)
    model.train()

    for epoch in range(args.train.epochs):
        print('======= epoch {}/{} ======='.format(epoch, args.train.epochs))
        loss_epch = 0.
        loss_mss_epch = 0.
        loss_f0_epch = 0.
        loss_krtss_epch = 0.
        loss_jttr_epch = 0.
        loss_shmmr_epch = 0.
        ctr_stime = time.time()
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad(set_to_none=True)

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()

            # forward
            signal, f0_pred, _, _ = model(mel=data['mel'], mel12=data['mel12'], f0_norm=data['norm_f0'])

            # loss
            loss, (loss_mss, loss_f0, loss_kurtosis, loss_jitter, loss_shimmer) = loss_func(
                signal, data['audio'], f0_pred, data['f0'], is_val=False)

            loss_epch += loss
            loss_mss_epch += loss_mss
            loss_f0_epch += loss_f0
            loss_krtss_epch += loss_kurtosis
            loss_jttr_epch += loss_jitter
            loss_shmmr_epch += loss_shimmer

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
                    'Batch {}/{} | {} | train loss: {:.6f}, mss loss: {:.6f}, F0 loss: {:.6f} | iteration: {} | avg. time: {:.2f}s'.format(
                        batch_idx,
                        num_batches,
                        saver.expdir,
                        loss.item(),
                        loss_mss.item(),
                        loss_f0.item(),
                        saver.global_step,
                        time_taken/args.train.interval_log
                    )
                )
                ctr_stime = time.time()  # when epoch doesnt end

        test_loss, test_loss_mss, test_loss_krtss, test_loss_f0, test_loss_jttr, test_loss_shmmr, time_taken = infer_on_validation(
            args, model, loss_func, loader_test,
            path_gendir=os.path.join(args.env.expdir, str(epoch)),
            generate_flag=generate_files,
            is_part=is_part,
            epoch=epoch)
        print(
            'epoch {}/{} val loss: {:.6f}, mss loss: {:.6f}, F0 loss: {:.6f} | time: {:.2f}'.format(
                epoch,
                args.train.epochs,
                loss.item(),
                loss_mss.item(),
                loss_f0.item(),
                time_taken
            )
        )
        test_loss_to_save = test_loss_mss + test_loss_f0

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(metrics=test_loss_to_save)
        after_lr = optimizer.param_groups[0]["lr"]
        print("AdamW lr %.4f -> %.4f" % (before_lr, after_lr))

        # log in tensorboard
        train_writer.add_scalar("Loss", loss_epch/num_batches, epoch)
        train_writer.add_scalar("MSS Loss", loss_mss_epch/num_batches, epoch)
        train_writer.add_scalar("F0 Loss", loss_f0_epch/num_batches, epoch)
        train_writer.add_scalar("Kurtosis Loss", loss_krtss_epch/num_batches, epoch)
        train_writer.add_scalar("Jitter Loss", loss_jttr_epch/num_batches, epoch)
        train_writer.add_scalar("Shimmer Loss", loss_shmmr_epch/num_batches, epoch)
        train_writer.add_scalar("LR", before_lr, epoch)
        train_writer.flush()

        val_writer.add_scalar("Loss", test_loss, epoch)
        val_writer.add_scalar("MSS Loss", test_loss_mss, epoch)
        val_writer.add_scalar("F0 Loss", test_loss_f0, epoch)
        val_writer.add_scalar("Kurtosis Loss", test_loss_krtss, epoch)
        val_writer.add_scalar("Jitter Loss", test_loss_jttr, epoch)
        val_writer.add_scalar("Shimmer Loss", test_loss_shmmr, epoch)
        val_writer.flush()

        #change to training mode
        model.train()

        # save best model
        if test_loss_to_save < best_loss:
            saver.save_models(
                {'vocoder': model}, postfix='best')
            best_loss = test_loss_to_save

    train_writer.close()
    val_writer.close()



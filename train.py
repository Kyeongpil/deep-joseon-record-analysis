import argparse
import gc
import logging
import os
import pickle
import time
import warnings

import torch
import torch.distributed as dist
from mpi4py import MPI
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model.dataset import HanjaDataset, HanjaKoreanDataset
from model.optimizer import Ralamb, get_cosine_schedule_with_warmup
from model.transformer import Transformer
from utils import TqdmLoggingHandler, accuracy, label_smoothing_loss, set_seed, write_log

warnings.simplefilter("ignore", UserWarning)


def main(args):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    device = torch.device("cuda")

    logger = None
    tb_logger = None
    if rank == 0:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(args.tensorboard_log_dir):
            os.mkdir(args.tensorboard_log_dir)
        tb_logger = SummaryWriter(f"{args.tensorboard_log_dir}/{args.model_name}")

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = TqdmLoggingHandler()
        handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    write_log(logger, "Load data")
    def load_data(args):
        gc.disable()
        with open(f"{args.preprocessed_data_path}/hanja_korean_word2id.pkl", "rb") as f:
            data = pickle.load(f)
            hanja_word2id = data['hanja_word2id']
            korean_word2id = data['korean_word2id']
        
        with open(f"{args.preprocessed_data_path}/preprocessed_train.pkl", "rb") as f:
            data = pickle.load(f)
            train_hanja_indices = data['hanja_indices']
            train_korean_indices = data['korean_indices']
            train_additional_hanja_indices = data['additional_hanja_indices']

        with open(f"{args.preprocessed_data_path}/preprocessed_valid.pkl", "rb") as f:
            data = pickle.load(f)
            valid_hanja_indices = data['hanja_indices']
            valid_korean_indices = data['korean_indices']
            valid_additional_hanja_indices = data['additional_hanja_indices']

        gc.enable()
        write_log(logger, "Finished loading data!")
        return (
            hanja_word2id, korean_word2id, 
            train_hanja_indices, train_korean_indices, train_additional_hanja_indices,
            valid_hanja_indices, valid_korean_indices, valid_additional_hanja_indices
        )

    # load data
    (
        hanja_word2id, korean_word2id, 
        train_hanja_indices, train_korean_indices, train_additional_hanja_indices,
        valid_hanja_indices, valid_korean_indices, valid_additional_hanja_indices
    ) = load_data(args)
    hanja_vocab_num = len(hanja_word2id)
    korean_vocab_num = len(korean_word2id)

    hk_dataset = HanjaKoreanDataset(train_hanja_indices, train_korean_indices, 
                    min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    hk_sampler = DistributedSampler(hk_dataset, num_replicas=world_size, rank=rank)
    hk_loader = DataLoader(hk_dataset, drop_last=True, batch_size=args.hk_batch_size, 
                    sampler=hk_sampler, num_workers=args.num_workers, prefetch_factor=4, pin_memory=True)
    write_log(logger, f"hanja-korean: {len(hk_dataset)}, {len(hk_loader)}")

    h_dataset = HanjaDataset(train_hanja_indices, train_additional_hanja_indices, hanja_word2id, 
                    min_len=args.min_len, src_max_len=args.src_max_len)
    h_sampler = DistributedSampler(h_dataset, num_replicas=world_size, rank=rank)
    h_loader = DataLoader(h_dataset, drop_last=True, batch_size=args.h_batch_size, 
                    sampler=h_sampler, num_workers=args.num_workers, prefetch_factor=4, pin_memory=True)
    write_log(logger, f"hanja: {len(h_dataset)}, {len(h_loader)}")

    hk_valid_dataset = HanjaKoreanDataset(valid_hanja_indices, valid_korean_indices, 
                        min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    hk_valid_sampler = DistributedSampler(hk_valid_dataset, num_replicas=world_size, rank=rank)
    hk_valid_loader = DataLoader(hk_valid_dataset, drop_last=True, batch_size=args.hk_batch_size, 
                        sampler=hk_valid_sampler)
    write_log(logger, f"hanja-korean-valid: {len(hk_valid_dataset)}, {len(hk_valid_loader)}")

    h_valid_dataset = HanjaDataset(valid_hanja_indices, valid_additional_hanja_indices, hanja_word2id, 
                    min_len=args.min_len, src_max_len=args.src_max_len)
    h_valid_sampler = DistributedSampler(h_valid_dataset, num_replicas=world_size, rank=rank)
    h_valid_loader = DataLoader(h_valid_dataset, drop_last=True, batch_size=args.h_batch_size, 
                        sampler=h_valid_sampler)
    write_log(logger, f"hanja: {len(h_valid_dataset)}, {len(h_valid_loader)}")

    del (
        train_hanja_indices, train_korean_indices, train_additional_hanja_indices,
        valid_hanja_indices, valid_korean_indices, valid_additional_hanja_indices
    )

    write_log(logger, "Build model")
    model = Transformer(hanja_vocab_num, korean_vocab_num, pad_idx=args.pad_idx, bos_idx=args.bos_idx,
                eos_idx=args.eos_idx, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len, 
                d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head, dropout=args.dropout,
                dim_feedforward=args.dim_feedforward, num_encoder_layer=args.num_encoder_layer, 
                num_decoder_layer=args.num_decoder_layer, num_mask_layer=args.num_mask_layer).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    for param in model.parameters():
        dist.broadcast(param.data, 0)

    dist.barrier()
    write_log(logger, f"Total Parameters: {sum([p.nelement() for p in model.parameters()])}")

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = Ralamb(params=optimizer_grouped_parameters, lr=args.lr)

    total_iters = round(len(hk_loader)/args.num_grad_accumulate*args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(total_iters*args.warmup_ratio), total_iters)
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        def load_states():
            checkpoint = torch.load(f'{args.save_path}/{args.model_name}_ckpt.pt', map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            return start_epoch

        start_epoch = load_states()

    write_log(logger, f"Training start - Total iter: {total_iters}\n")
    iter_num = round(len(hk_loader)/args.num_grad_accumulate)
    global_step = start_epoch*iter_num
    hk_iter = iter(hk_loader)
    h_iter = iter(h_loader)
    model.train()
    tgt_mask = Transformer.generate_square_subsequent_mask(args.trg_max_len - 1, device)
    
    # validation
    validate(model, tgt_mask, h_valid_loader, hk_valid_loader, rank, logger, tb_logger, 0, device)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        while True:
            start = time.time()
            finish_epoch = False
            trans_top5, trans_loss, mask_top5, mask_loss = 0.0, 0.0, 0.0, 0.0

            if args.train_reconstruct:
                optimizer.zero_grad(set_to_none=True)
                for _ in range(args.num_grad_accumulate):
                    try:
                        src_sequences, trg_sequences = next(h_iter)
                    except StopIteration:
                        h_sampler.set_epoch(epoch)
                        h_iter = iter(h_loader)
                        src_sequences, trg_sequences = next(h_iter)

                    trg_sequences = trg_sequences.to(device)
                    src_sequences = src_sequences.to(device)
                    non_pad = trg_sequences != args.pad_idx
                    trg_sequences = trg_sequences[non_pad].contiguous().view(-1)

                    with autocast():
                        predicted = model.module.reconstruct_predict(
                            src_sequences, masked_position=non_pad)
                        predicted = predicted.view(-1, predicted.size(-1))
                        loss = label_smoothing_loss(predicted, trg_sequences)/args.num_grad_accumulate

                    scaler.scale(loss).backward()

                    if global_step % args.print_freq == 0:
                        mask_top5 += accuracy(predicted, trg_sequences, 5)/args.num_grad_accumulate
                        mask_loss += loss.detach().item()

                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data = param.grad.data / world_size

                scaler.step(optimizer)
                scaler.update()

            if args.train_translate:
                optimizer.zero_grad(set_to_none=True)
                for _ in range(args.num_grad_accumulate):
                    try:
                        src_sequences, trg_sequences = next(hk_iter)
                    except StopIteration:
                        hk_sampler.set_epoch(epoch)
                        hk_iter = iter(hk_loader)
                        src_sequences, trg_sequences = next(hk_iter)
                        finish_epoch = True

                    trg_sequences = trg_sequences.to(device)
                    trg_sequences_target = trg_sequences[:, 1:]
                    src_sequences = src_sequences.to(device)
                    non_pad = trg_sequences_target != args.pad_idx
                    trg_sequences_target = trg_sequences_target[non_pad].contiguous().view(-1)

                    with autocast():
                        predicted = model(
                            src_sequences, trg_sequences[:, :-1], tgt_mask, non_pad_position=non_pad)
                        predicted = predicted.view(-1, predicted.size(-1))
                        loss = label_smoothing_loss(predicted, trg_sequences_target) / args.num_grad_accumulate

                    scaler.scale(loss).backward()

                    if global_step % args.print_freq == 0:
                        trans_top5 += accuracy(predicted, trg_sequences_target, 5)/args.num_grad_accumulate
                        trans_loss += loss.detach().item()

                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data = param.grad.data / world_size

                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            # Print status
            if global_step % args.print_freq == 0:
                if args.train_reconstruct:
                    mask_top5 = torch.cuda.FloatTensor([mask_top5])
                    mask_loss = torch.cuda.FloatTensor([mask_loss])
                    dist.all_reduce(mask_top5, op=dist.ReduceOp.SUM)
                    dist.all_reduce(mask_loss, op=dist.ReduceOp.SUM)
                    mask_top5 = (mask_top5 / world_size).item()
                    mask_loss = (mask_loss / world_size).item()

                if args.train_translate:
                    trans_top5 = torch.cuda.FloatTensor([trans_top5])
                    trans_loss = torch.cuda.FloatTensor([trans_loss])
                    dist.all_reduce(trans_top5, op=dist.ReduceOp.SUM)
                    dist.all_reduce(trans_loss, op=dist.ReduceOp.SUM)
                    trans_top5 = (trans_top5 / world_size).item()
                    trans_loss = (trans_loss / world_size).item()

                if rank == 0:
                    batch_time = time.time() - start
                    write_log(
                        logger, 
                        f'[{global_step}/{total_iters}, {epoch}]\tIter time: {batch_time:.3f}\t'
                        f'Trans loss: {trans_loss:.3f}\tMask_loss: {mask_loss:.3f}\t'
                        f'Trans@5: {trans_top5:.3f}\tMask@5: {mask_top5:.3f}')

                    tb_logger.add_scalar('loss/translate', trans_loss, global_step)
                    tb_logger.add_scalar('loss/mask', mask_loss, global_step)
                    tb_logger.add_scalar('top5/translate', trans_top5, global_step)
                    tb_logger.add_scalar('top5/mask', mask_top5, global_step)
                    tb_logger.add_scalar('batch/time', batch_time, global_step)
                    tb_logger.add_scalar('batch/lr', optimizer.param_groups[0]['lr'], global_step)

            global_step += 1
            if finish_epoch:
                break

        # validation
        validate(model, tgt_mask, h_valid_loader, hk_valid_loader, rank, logger, tb_logger, epoch, device)
        # save model
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, f'{args.save_path}/{args.model_name}_ckpt.pt')
            write_log(logger, f"***** {epoch}th model updated! *****")


def validate(model, tgt_mask, h_valid_loader, hk_valid_loader, rank, logger, tb_logger, epoch, device):
    model.eval()
    trans_loss, trans_accuracy, mask_loss, mask_accuracy = 0.0, 0.0, 0.0, 0.0
    if args.train_reconstruct:
        for src_sequences, trg_sequences in h_valid_loader:
            src_sequences = src_sequences.to(device)
            trg_sequences = trg_sequences.to(device)

            masked_position = trg_sequences != args.pad_idx
            trg_sequences = trg_sequences[masked_position].contiguous().view(-1)

            with torch.no_grad():
                batch_size = src_sequences.size(0)
                predicted = model.module.reconstruct_predict(src_sequences, masked_position=masked_position)
                predicted = predicted.view(-1, predicted.size(-1))
                mask_loss += label_smoothing_loss(predicted, trg_sequences) * batch_size
                mask_accuracy += accuracy(predicted, trg_sequences, 5) * batch_size

        mask_accuracy = torch.cuda.FloatTensor([mask_accuracy])
        mask_loss = torch.cuda.FloatTensor([mask_loss])
        dist.all_reduce(mask_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(mask_loss, op=dist.ReduceOp.SUM)
        mask_accuracy = (mask_accuracy / len(h_valid_loader.dataset)).item()
        mask_loss = (mask_loss / len(h_valid_loader.dataset)).item()

    if args.train_translate:
        for src_sequences, trg_sequences in hk_valid_loader:
            src_sequences = src_sequences.to(device)
            trg_sequences = trg_sequences.to(device)
            trg_sequences_target = trg_sequences[:, 1:]
            non_pad = trg_sequences_target != args.pad_idx
            trg_sequences_target = trg_sequences_target[non_pad].contiguous().view(-1)

            with torch.no_grad():
                batch_size = src_sequences.size(0)
                predicted = model.module.forward(
                    src_sequences, trg_sequences[:, :-1], tgt_mask, non_pad_position=non_pad)
                predicted = predicted.view(-1, predicted.size(-1))
                trans_loss += label_smoothing_loss(predicted, trg_sequences_target) * batch_size
                trans_accuracy += accuracy(predicted, trg_sequences_target, 5) * batch_size

        trans_accuracy = torch.cuda.FloatTensor([trans_accuracy])
        trans_loss = torch.cuda.FloatTensor([trans_loss])
        dist.all_reduce(trans_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(trans_loss, op=dist.ReduceOp.SUM)
        trans_accuracy = (trans_accuracy / len(hk_valid_loader.dataset)).item()
        trans_loss = (trans_loss / len(hk_valid_loader.dataset)).item()

    if rank == 0:
        write_log(logger, f'Epoch: [{epoch}]\tTrans loss: {trans_loss:.4f}\t Mask loss: {mask_loss:.4f}')
        tb_logger.add_scalar('valid_loss/translate_loss', trans_loss, epoch)
        tb_logger.add_scalar('valid_loss/mask_loss', mask_loss, epoch)
        tb_logger.add_scalar('valid_accuracy/trans_accuracy', trans_accuracy, epoch)
        tb_logger.add_scalar('valid_accuracy/mask_accuracy', mask_accuracy, epoch)

    model.train()
    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train machine translation.')
    parser.add_argument('--preprocessed_data_path',
        default='/home/nas1_userC/rudvlf0413/joseon_translation/dataset/preprocessed', type=str)
    parser.add_argument('--save_path', default='./models', type=str)
    parser.add_argument('--tensorboard_log_dir', default='./tensor_logs', type=str)

    parser.add_argument('--min_len', default=3, type=int)
    parser.add_argument('--src_max_len', default=300, type=int, help='max length of the source sentence')
    parser.add_argument('--trg_max_len', default=350, type=int, help='max length of the target sentence')
    parser.add_argument('--pad_idx', default=0, type=int)
    parser.add_argument('--bos_idx', default=1, type=int)
    parser.add_argument('--eos_idx', default=2, type=int)
    
    parser.add_argument('--d_model', default=768, type=int)
    parser.add_argument('--d_embedding', default=256, type=int)
    parser.add_argument('--n_head', default=12, type=int)
    parser.add_argument('--dim_feedforward', default=3072, type=int)
    parser.add_argument('--num_encoder_layer', default=12, type=int)
    parser.add_argument('--num_mask_layer', default=6, type=int)
    parser.add_argument('--num_decoder_layer', default=12, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.3, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--hk_batch_size', default=10, type=int, help='batch size for hanja, korean')
    parser.add_argument('--h_batch_size', default=16, type=int, help='batch size for hanja')
    parser.add_argument('--num_grad_accumulate', default=8, type=int)

    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--master_port', default=9996, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--train_reconstruct', default=True, type=bool)
    parser.add_argument('--train_translate', default=True, type=bool)

    args = parser.parse_args()
    args.model_name = f"model_test_{args.num_encoder_layer}_{args.num_mask_layer}_{args.num_decoder_layer}"
    set_seed(args.seed)

    main(args)

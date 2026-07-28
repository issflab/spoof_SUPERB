import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
import yaml
from spoof_superb.data.datasets_ssl import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from spoof_superb.models.aasist import Model as aasist_model
# from sls_model import Model as sls_model
from spoof_superb.models.linear_head import UtteranceLevel as LinearHead
from spoof_superb.models.aasist_raw import Model as aasist_raw_model
from spoof_superb.config import cfg
from spoof_superb.core.utils import create_optimizer, seed_worker, set_seed, str_to_bool
from spoof_superb.core.metrics import calculate_EER

from s3prl import hub

from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

from sklearn.metrics import balanced_accuracy_score
import json
from tqdm import tqdm

__author__ = "Hashim Ali"
__email__ = "alhashim@umich.edu"


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "configs" / "AASIST.conf"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
LOGS_ROOT = OUTPUTS_ROOT / "logs"
METRICS_ROOT = OUTPUTS_ROOT / "models"


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    y_true = []
    y_pred = []

    for batch_x, utt_id, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        batch_score = batch_score.tolist()
        
        pred = ["fake" if bs < 0 else "bonafide" for bs in batch_score]
        keys = ["fake" if by == 0 else "bonafide" for by in batch_y.tolist()]
        y_pred.extend(pred)
        y_true.extend(keys)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
   
    return val_loss, balanced_acc


def produce_evaluation(data_loader, model, device, save_path, trial_path):
    model.eval()

    val_loss = 0.0
    num_total = 0.0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    fname_list = []
    key_list = []
    score_list = []

    trial_lines = []

    with open(trial_path, "r") as f_trl:
        trial_lines.extend(f_trl.readlines())
    
    for batch_x, utt_id, batch_y in tqdm(data_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() 
        batch_score = batch_score.tolist()

        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
        # Writing score to file
        fname_list.extend(utt_id)
        score_list.extend(batch_score)
        key_list.extend(batch_y)
    
    print(len(fname_list), len(key_list), len(score_list), len(trial_lines))
    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')

            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    
    # with open(save_path, 'w') as fh:
    #     for f, k, cm in zip(fname_list, key_list, score_list):
    #         fh.write('{} {} {}\n'.format(f, k, cm))
    # fh.close()  

    val_loss /= num_total

    print('Scores saved to {}'.format(save_path))

    return val_loss


def train_epoch(train_loader, model, optimizer, device, accum_steps=1):
    """Train one epoch.

    accum_steps > 1 accumulates gradients over that many loader batches before
    stepping, so the EFFECTIVE batch size is loader_batch * accum_steps. This
    exists for aasist_raw: at raw-waveform resolution its first residual block
    holds a (B, 32, 24, 21490) activation -- 4.2 GB at B=64 -- so the benchmark's
    batch size of 64 cannot fit on a 40 GB A100, while the SSL models can
    because their upstream downsamples to ~200 frames first. Accumulation keeps
    the optimisation identical to the SSL recipe (same effective batch, same LR,
    same number of optimizer steps) instead of quietly training at a smaller
    batch.

    accum_steps=1 runs the original loop verbatim (bit-identical weights), so
    the SSL architectures are unaffected.

    The accumulation path cannot simply divide each micro-batch loss by
    accum_steps: this benchmark's loss is CrossEntropyLoss with class weights
    [0.1, 0.9] and reduction='mean', which normalises by the SUM OF SAMPLE
    WEIGHTS, not by the sample count. Averaging four such means only equals the
    batch-64 mean if every micro-batch has identical class composition, which is
    false for shuffled data. Instead the micro-batches accumulate the weighted
    SUM and the group's gradients are divided by the group's total sample weight
    just before stepping -- algebraically exactly the batch-64 update.
    (tests/test_grad_accum.py pins both properties.)
    """
    running_loss = 0

    num_total = 0.0

    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    if accum_steps == 1:
        for batch_x, utt_id, batch_y in tqdm(train_loader):

            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)

            batch_loss = criterion(batch_out, batch_y)

            running_loss += (batch_loss.item() * batch_size)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        running_loss /= num_total

        return running_loss

    criterion_sum = nn.CrossEntropyLoss(weight=weight, reduction='sum')

    optimizer.zero_grad()
    n_pending = 0
    group_weight = 0.0

    for batch_x, utt_id, batch_y in tqdm(train_loader):

        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)

        loss_sum = criterion_sum(batch_out, batch_y)
        batch_weight = weight[batch_y].sum()

        # report the same quantity the accum_steps=1 branch reports
        running_loss += (loss_sum.item() / batch_weight.item()) * batch_size

        loss_sum.backward()
        group_weight += batch_weight.item()
        n_pending += 1

        if n_pending == accum_steps:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= group_weight
            optimizer.step()
            optimizer.zero_grad()
            n_pending, group_weight = 0, 0.0

    # flush a trailing partial accumulation group
    if n_pending:
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= group_weight
        optimizer.step()
        optimizer.zero_grad()

    running_loss /= num_total

    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSL + AASIST System trained on multiple datasets')

    parser.add_argument('--database_path', type=str, default='/data/Data/ASVSpoofData_2019/train/LA/', help='Change this to the base data directory which contains datasets.')
    parser.add_argument('--protocols_path', type=str, default='/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/', help='Change this to the path which contain protocol files')
    
    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('--ssl_model', type=str, default='wavlm_large', help='''Change this to the ssl model name.
                        Available options in S3PRL: {}.
                        '''.format(upstreams))

    # Hyperparameters
    parser.add_argument('--model_arch', type=str, default=None,
                        choices=['aasist', 'sls', 'linear_head', 'aasist_raw', 'lfcc_gmm'],
                        help='Backend architecture. Previously settable only via the '
                             'SSL_MODEL_ARCH environment variable.')
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'eval'],
                        help='Previously settable only via SSL_MODE.')
    parser.add_argument('--train_dataset', type=str, default=None,
                        help='Training-set tag used to name the checkpoint '
                             '(cfg.dataset). Previously settable only via SSL_DATASET.')
    parser.add_argument('--batch_size', type=int, default=14)
    # Effective batch stays --batch_size; this only splits it into smaller
    # forward/backward chunks so a memory-heavy model fits. 0 = disabled
    # (single chunk), which is the existing behaviour for the SSL runs.
    parser.add_argument('--micro_batch', type=int, default=0,
                        help='Split each batch into chunks of this size and '
                             'accumulate gradients (effective batch is unchanged). '
                             'Needed for aasist_raw, which cannot fit batch 64 at '
                             'raw-waveform resolution.')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    # Baselines only (aasist_raw / lfcc_gmm): which benchmark set to score.
    # Ignored by the SSL architectures, which keep their existing eval path.
    parser.add_argument('--eval_dataset', type=str, default=None,
                        help='Benchmark dataset key for the non-SSL baselines, '
                             'e.g. eval_2019, asvspoof2021_DF, wild. '
                             'See `python -m spoof_superb.scoring.driver --list_datasets`.')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Worker processes for LFCC feature extraction (lfcc_gmm only).')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)')
    

    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in percentage.[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    # train_protocol_filename = 'SAFE_challenge_train_latest_protocol.txt'
    # dev_protocol_filename = 'SAFE_challenge_dev_latest_protocol.txt'

    # train_protocol_filename = 'SAFE_challenge_train_latest_protocol.txt'
    # dev_protocol_filename = 'SAFE_challenge_dev_latest_protocol.txt'

    # train_protocol_filename = 'SAFE_Challenge_train_protocol_Codec_FF_ITW_Pod_mlaad_spoofceleb.txt'
    # dev_protocol_filename = 'SAFE_Challenge_dev_protocol_Codec_FF_ITW_Pod_mlaad_spoofceleb.txt'
    
    args = parser.parse_args()

    # Override config with CLI if provided
    if args.database_path:
        cfg.database_path = args.database_path
    if args.protocols_path:
        cfg.protocols_path = args.protocols_path
    if args.model_path:
        cfg.pretrained_checkpoint = args.model_path
    if args.model_arch:
        cfg.model_arch = args.model_arch
    if args.mode:
        cfg.mode = args.mode
    if args.train_dataset:
        cfg.dataset = args.train_dataset

    with open(CONFIG_PATH, "r") as f_json:
        args_config = json.loads(f_json.read())

    optim_config = args_config["optim_config"]
    optim_config["epochs"] = args.num_epochs

    #make experiment reproducible
    set_random_seed(args.seed, args)

    #define model saving path
    # The non-SSL baselines take no upstream; recording --ssl_model's default
    # in their checkpoint path would be misleading. SSL tags are unchanged.
    tag_ssl = 'none' if cfg.model_arch in ('lfcc_gmm', 'aasist_raw') else args.ssl_model
    model_tag = 'model_{}_{}_{}_{}_{}_{}'.format(args.loss, args.num_epochs, args.batch_size, cfg.model_arch, cfg.dataset, tag_ssl)
    
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    
    # config.py no longer creates directories as an import side effect.
    cfg.prepare_dirs()
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(LOGS_ROOT / model_tag))

    # prepare save path
    model_save_path = os.path.join(cfg.save_dir, model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    

    #GPU device
    device = cfg.cuda_device if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # ------------------------------------------------------------------ #
    # Non-SSL baselines: dispatch out of the SSL-shaped pipeline.
    #
    # lfcc_gmm has no gradients, no DataLoader and no GPU kernel -- it is EM
    # over two diagonal GMMs. Running it through the loop below would be
    # theatre, so main.py hands it to its own trainer and exits with that
    # status. Evaluation for both baselines goes to spoof_superb.scoring.driver,
    # the single scoring entry point that replaced the three standalone
    # eval_*.py drivers (main.py's eval branch below
    # reads cfg.eval_protocol, which has no default and no CLI setter, and
    # Dataset_ASVspoof2021_eval hardcodes release_in_the_wild/*.wav).
    #
    # Everything above this point is shared with the SSL runs; nothing below
    # it is reached for the baselines, so the SSL paths are untouched.
    # ------------------------------------------------------------------ #
    if cfg.model_arch in ('lfcc_gmm', 'aasist_raw'):
        if cfg.mode == 'eval':
            from spoof_superb.scoring.driver import run_eval_from_main
            raise SystemExit(run_eval_from_main(args, cfg, device))
        if cfg.model_arch == 'lfcc_gmm':
            from spoof_superb.train.lfcc_gmm import run_train_from_main
            raise SystemExit(run_train_from_main(args, cfg))
        # aasist_raw in train mode falls through to the shared torch loop.

    # instantiate model based on config
    if cfg.model_arch == 'aasist':
        model = aasist_model(args, device)
    elif cfg.model_arch == 'sls':
        model = sls_model(args, device)
    elif cfg.model_arch == 'linear_head':
        model = LinearHead(args, device)
    elif cfg.model_arch == 'aasist_raw':
        # Non-SSL baseline: sinc front-end + AASIST graph back-end. Trains
        # through the identical loop below (same loss, optimizer, scheduler,
        # RawBoost augmentation and dev-EER checkpoint selection as the SSL
        # runs); the only difference is that no s3prl upstream is built.
        model = aasist_raw_model(args, device)
    else:
        raise ValueError(f'Unknown model architecture: {cfg.model_arch}')
    model = model.to(device)

    nb_params = sum(p.numel() for p in model.parameters())
    print('nb_params:', nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    # load pretrained checkpoint if given
    if cfg.pretrained_checkpoint:
        print('Loading pretrained checkpoint from', cfg.pretrained_checkpoint)
        model.load_state_dict(torch.load(cfg.pretrained_checkpoint, map_location=device))

    # dataset prep
    train_proto = os.path.join(cfg.protocols_path, cfg.train_protocol)
    dev_proto = os.path.join(cfg.protocols_path, cfg.dev_protocol)

    d_label_trn, file_train = genSpoof_list(train_proto, is_train=True)
    d_label_dev, file_dev = genSpoof_list(dev_proto, is_train=False)

    print("Train protocol:", getattr(cfg, "train_protocol_path", cfg.train_protocol))
    print("Dev protocol:", getattr(cfg, "dev_protocol_path", cfg.dev_protocol))
    print("Database base path:", cfg.database_path)
    # for tag, file_list in (("TRAIN", file_train), ("DEV", file_dev)):
    #     print(f"Sample {tag} utt_ids (first 3):", file_list[:3])
    #     for utt in file_list[:3]:
    #         full = os.path.join(cfg.database_path, utt + '.flac')
    #         print(f"  [{tag}] {utt} -> {full} exists: {os.path.isfile(full)}")
    

    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn,
                                            base_dir=os.path.join(cfg.database_path, 'ASVspoof2019_LA_train'), 
                                            algo=args.algo)
    dev_set = Dataset_ASVspoof2019_train(args, list_IDs=file_dev, labels=d_label_dev,
                                            base_dir=os.path.join(cfg.database_path, 'ASVspoof2019_LA_dev'), algo=args.algo)

    print('no. of training trials', len(file_train))
    print('no. of validation trials', len(file_dev))

    # micro_batch=0 keeps the original single-chunk behaviour (loader batch ==
    # args.batch_size, accum_steps == 1); anything else preserves the same
    # effective batch via gradient accumulation. See train_epoch.
    if args.micro_batch and args.micro_batch < args.batch_size:
        loader_batch = args.micro_batch
        accum_steps = max(1, args.batch_size // args.micro_batch)
    else:
        loader_batch = args.batch_size
        accum_steps = 1
    print(f'loader batch: {loader_batch}, accum steps: {accum_steps}, '
          f'effective batch: {loader_batch * accum_steps}')

    train_loader = DataLoader(train_set, batch_size=loader_batch,
                              num_workers=8, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=loader_batch,
                            num_workers=8, shuffle=False)

    # optimizer + scheduler
    # load external config if exists
    with open(CONFIG_PATH, "r") as f_json:
        args_config = json.loads(f_json.read())
    optim_config = args_config["optim_config"]
    optim_config["epochs"] = args.num_epochs
    optim_config["steps_per_epoch"] = len(train_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # make directory for metric logging
    metric_path = METRICS_ROOT / model_tag / "metrics"
    metric_path.mkdir(parents=True, exist_ok=True)

    os.path.join(str(metric_path), "dev_score.txt")
    # train vs eval
    if cfg.mode == 'train':
        # BUG (pre-existing, left in place for the SSL architectures): this
        # initial value is compared against calculate_EER(), which returns a
        # PERCENTAGE (e.g. 24.53), not a fraction. With best_val_eer = 1 no
        # checkpoint is written and no SWA update happens until dev EER drops
        # below 1%, so a model that never gets there finishes 50 epochs and
        # saves nothing at all -- neither epoch_*.pth nor swa.pth.
        #
        # Only the non-SSL baselines are corrected here, to honour the
        # "do not change existing SSL behaviour" constraint. This needs a
        # global fix (and probably a re-run of any SSL model whose dev EER
        # never went below 1%) -- see humanpending.md.
        if cfg.model_arch in ('aasist_raw', 'lfcc_gmm'):
            best_val_eer = float('inf')
        else:
            best_val_eer = 1
        n_swa_update = 0

        for epoch in range(args.num_epochs):
            train_loss = train_epoch(train_loader, model, optimizer, device,
                                     accum_steps=accum_steps)

            val_loss = produce_evaluation(
                dev_loader,
                model,
                device,
                os.path.join(str(metric_path), "dev_score.txt"),
                dev_proto,
            )

            dev_eer = calculate_EER(cm_scores_file=os.path.join(str(metric_path), "dev_score.txt"))

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_eer', dev_eer, epoch)
            print(f"Epoch {epoch} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_eer: {dev_eer:.4f}")

            if dev_eer < best_val_eer:
                print(f"Best model updated at epoch {epoch}")
                best_val_eer = dev_eer
                torch.save(model.state_dict(),
                           os.path.join(model_save_path, "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer)))
                
                # SWA update on improvement (as per prior logic)
                print("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1

            writer.add_scalar("best_dev_eer", best_val_eer, epoch)

        print("Finalizing SWA (if any updates occurred)")
        if n_swa_update > 0:
            optimizer_swa.swap_swa_sgd()
            optimizer_swa.bn_update(train_loader, model, device=device)
            torch.save(model.state_dict(), os.path.join(model_save_path, "swa.pth"))

    elif cfg.mode == 'eval':

        eval_proto = os.path.join(cfg.protocols_path, cfg.eval_protocol)
        
        file_eval = genSpoof_list(eval_proto, is_train=False, is_eval=True)
        eval_set = Dataset_ASVspoof2021_eval(args, list_IDs=file_eval, base_dir=os.path.join(cfg.database_path, 'ASVspoof2019_LA_eval'))

        print('no. of eval trials',len(file_eval))
        
        # fallback to best.pth if no explicit checkpoint
        if not cfg.pretrained_checkpoint:
            candidate = os.path.join(cfg.save_dir, cfg.model_name, 'best.pth')
            if os.path.isfile(candidate):
                print('Loading best checkpoint from', candidate)
                model.load_state_dict(torch.load(candidate, map_location=device))
        
        # val_loss, val_balanced_acc = evaluate_accuracy(eval_loader, model, device)
        # print(f'EVAL: val_loss={val_loss:.4f}, balanced_acc={val_balanced_acc:.4f}')

        produce_evaluation(eval_set, model, device, args.eval_output, eval_proto)

    else:
        raise ValueError("cfg.mode must be 'train' or 'eval'")
    
    

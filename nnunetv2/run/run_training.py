import multiprocessing
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1])) # 一般在程序开头设置
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
from torch.autograd import Variable
import numpy as np

def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda'),prefix='client0'):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    print('nnunet_trainer',nnunet_trainer)
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, unpack_dataset=not use_compressed, device=device, prefix=prefix)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                           use_compressed)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()

def update_global_model(net_clients_trainer, client_weight, client_list):
    # client_num=4
    # Use the true average until the exponential average is more correct
    
    for param in zip(*list(net_clients_trainer[i].network.parameters() for i in client_list)):
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda() 
        for i in range(len(client_list)):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(len(client_list)):
            param[i].data.mul_(0).add_(new_para.data)

def run_training(args,dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'
    nnunet_trainer=[]
    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    else:

        for i in range(args.client_num):
            nnunet_trainer.append(get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, use_compressed_data, device=device,prefix='client{}'.format(i)))
            if disable_checkpointing:
                nnunet_trainer[-1].disable_checkpointing = disable_checkpointing

            assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

            maybe_load_checkpoint(nnunet_trainer[-1], continue_training, only_run_validation, pretrained_weights)


        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True
        slice_num=[]#[0 for i in range(args.client_num)]
        client_list=list(range(args.client_num))
        client_list.pop(args.unseen_site)
        print('client_list',client_list)
        if not only_run_validation:
            for i in range(args.client_num):
                if i==args.unseen_site:
                    continue
                print('client:',i)
                current_trainer=nnunet_trainer[i]
                current_trainer.on_train_start()
                slice_num.append(current_trainer.dataset_len)
            client_weight = slice_num / np.sum(slice_num)
            print('client_weight:',client_weight)
            for epoch in range(0, args.num_epochs):
                for i in range(args.client_num):
                    if i==args.unseen_site:
                        continue
                    print('client:',i)
                    current_trainer=nnunet_trainer[i]
                    current_trainer.on_epoch_start()

                    current_trainer.on_train_epoch_start()
                    train_outputs = []
                    for batch_id in range(current_trainer.num_iterations_per_epoch):
                        train_outputs.append(current_trainer.train_step(next(current_trainer.dataloader_train)))
                    current_trainer.on_train_epoch_end(train_outputs)
                print('aggragating...')
                update_global_model(nnunet_trainer,client_weight,client_list)
                for i in range(args.client_num):
                    if i==args.unseen_site:
                        continue      
                    current_trainer=nnunet_trainer[i]     
                    print('client:',i)
                    with torch.no_grad():
                        current_trainer.on_validation_epoch_start()
                        val_outputs = []
                        for batch_id in range(current_trainer.num_val_iterations_per_epoch):
                            val_outputs.append(current_trainer.validation_step(next(current_trainer.dataloader_val)))
                        current_trainer.on_validation_epoch_end(val_outputs)         
                    current_trainer.on_epoch_end()
            for i in range(args.client_num):
                if i==args.unseen_site:
                    continue
                current_trainer=nnunet_trainer[i]
                current_trainer.on_train_end()
            # nnunet_trainer[0].run_training()

        # if val_with_best:
        #    nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        #nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('--device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    parser.add_argument('--client_num', type=int, default=4, help='client number')
    parser.add_argument('--unseen_site', type=int, default=0, help='client number')
    parser.add_argument('--num_epochs', type=int, default=50, help='client number')
    args = parser.parse_args()

    # assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    else:
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')#:{}'.format(args.device))
        # print(device)

    run_training(args, args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.use_compressed, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                 device=device)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # multiprocessing.set_start_method("spawn")
    run_training_entry()

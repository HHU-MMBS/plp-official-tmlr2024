import argparse
from pathlib import Path

# Local imports
import utils
from eval.ood_scores import available_metrics, available_norms
from loaders import check_dataset, get_embeds_path, get_default_path

def get_eval_args(notebook=False):
    parser = get_eval_args_parser()

    args = parser.parse_args() if not notebook else parser.parse_args("")
    return _check_args(args)


def get_eval_args_parser():
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[20], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.02, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--datapath', default=None, type=str, help="Path to dataset. Set automatically")
    head_group = parser.add_argument_group('head', 'Head parameters. Might get overridden by hp.json')
    head_group.add_argument('--out_dim', default=100, type=int)
    head_group.add_argument('--nlayers', default=2, type=int, help='Head layers')
    head_group.add_argument('--hidden_dim', default=512, type=int, help="Head's hidden dim")
    head_group.add_argument('--bottleneck_dim', default=256, type=int, help="Head's bottleneck dim")
    head_group.add_argument('--l2_norm', default=False, help="Whether to apply L2 norm after backbone")
    head_group.add_argument('--teacher_temp', default=0.1, type=float, help="Temperature for teacher")
    parser.add_argument('--embed_norm', default=False, type=utils.bool_flag,
                        help="Whether to normalize embeddings using precomputed mean and std")
    parser.add_argument('--dataset', type=check_dataset)
    parser.add_argument('--pseudo_labels', type=str, default=None, help="Path to pseudo labels")
    
    parser.add_argument('--out_dist', type=check_dataset, nargs='*')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')    
    
    # Classification and clustering evals
    parser.add_argument('--eval_knn_acc', default=False, type=utils.bool_flag,
                        help="""Calculate KNN scores. Disable this for large data sets""")
    parser.add_argument('--eval_cluster', default=False, type=utils.bool_flag, help="""Calculate clustering scores""")
    # OOD related evals
    parser.add_argument('--eval_ood', default=True, type=utils.bool_flag, help="""Calculate AUROC and FPR95 scores""")
    
    parser.add_argument('--ood_knn_ks', default=[1], nargs='+', type=int)
    parser.add_argument('--eval_ood_norm', default=True, type=utils.bool_flag, help="""Calculate OOD max norms""")
    parser.add_argument('--ood_norms', default=["l1", "softmax"], choices=available_norms(), nargs='+')
    parser.add_argument('--eval_ood_knn', default=True, type=utils.bool_flag, help="""Calculate KNN OOD scores""")
    parser.add_argument('--ood_knn_metrics', default=['temp-cos-sim'], choices=available_metrics(), nargs='+')
    parser.add_argument('--eval_ood_logits', default=True, type=utils.bool_flag, help="""Calculate OOD logits scores""")
    parser.add_argument('--ood_logits_temp', default=[1.0], type=float, nargs='+',
                        help="""Temperature for OOD logits scores""")
    parser.add_argument('--eval_ood_maha', type=utils.bool_flag, default=False,
                        help="""Calculate Mahalanobis OOD scores""")
    parser.add_argument('--arch', default=None, type=str, help='Architecture')
    parser.add_argument('--head', default=True, type=utils.bool_flag,
                        help="Whether to load the DINO head")
    parser.add_argument('--ckpt_folder', type=Path, help='Folder containing the checkpoints')
    parser.add_argument('--out_dir', type=Path,
                        help='Folder to save the results. If not specified, will be ckpt_folder')
    parser.add_argument('--no_save', action='store_true', default=False, help='Whether to save the results')
    parser.add_argument('--no_cache', action='store_true', default=False, help='Whether to cache backbone results')
    parser.add_argument('--ignore_hp_file', action='store_true', default=False, help='Whether to ignore hp.json')
    parser.add_argument('--tensorboard', type=utils.bool_flag, default=False, help='Whether to use tensorboard')
    parser.add_argument('--lin_eval', type=utils.bool_flag, default=False)
    parser.add_argument('--fast_eval', type=utils.bool_flag, default=True, help="Evaluates only last ckpt and lowest loss")
    parser.add_argument('--precomputed', type=utils.bool_flag, default=True, help="Use precomputed embeddings")
    return parser

def _check_args(args):
    ood_score_given = args.eval_ood_knn or args.eval_ood_norm or args.eval_ood_maha or args.eval_ood_logits
    if args.eval_ood and not ood_score_given:
        raise ValueError('Please specify at least one OOD evaluation metric')
    if args.head and args.ckpt_folder is None:
        raise ValueError('Please specify a checkpoint folder with --ckpt_folder')
    if args.head and args.embed_norm:
        print('Explicit embedding normalization will be disabled for evaluation with head')
        args.embed_norm = False

    if args.out_dir and args.no_save:
        raise ValueError('Please specify either --out_dir or --no_save')
    if not args.no_save and not args.out_dir and not args.ckpt_folder:
        raise ValueError('Please specify an output folder with --out_dir or --ckpt_folder')
    if not args.no_save and not args.out_dir and args.ckpt_folder:
        args.out_dir = args.ckpt_folder   
    if args.datapath is None:
        args.datapath = get_embeds_path() if args.precomputed else get_default_path(args.dataset)
    return args
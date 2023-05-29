import argparse
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="LanguageProcessors Arguments")
    parser.add_argument('--model', type=str, default="BiGRU_RMonoAttn",
                        choices=["BiGRU",
                                 "BiGRUrel", "BiGRUrel_rev", "BiGRUrel_mixdir",
                                 "BiGRUrope", "BiGRUrope_mixdir",
                                 "BiGRU_locattn", "BiGRU_locattn_simple",
                                 "BiGRU_mixattn", "BiGRU_mixattn_simple", "BiGRU_mixattn_simplePR",
                                 "BiGRU_grumixattn", "BiGRU_gruxmixattn_simple",
                                 "BiGRU_OneStep", "BiGRU_MixOneStep", "BiGRU_MixOneStepPR",
                                 "BiGRU_FreeStep", "BiGRU_AblationStep", "BiGRU_SoftStairStep",
                                 "BiGRU_MonoAttn", "BiGRU_MixMonoAttn", "BiGRU_MixMonoAttnPR",
                                 "BiGRU_RMonoAttn", "BiGRU_MixRMonoAttn", "BiGRU_MixRMonoAttnPR"])
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="seq2seq",
                        choices=["seq2seq"])
    parser.add_argument('--dataset', type=str, default="copy",
                        choices=["llt", "rllt",
                                 "copy", "rc",
                                 "fpc", "rpc",
                                 "fgc", "rgc",
                                 "dedup", "posretrieve",
                                 "scan_length",
                                 "cfq_length"])
    parser.add_argument('--times', type=int, default=5)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--load_checkpoint', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--reproducible', type=str2bool, default=True, const=True, nargs='?')
    return parser

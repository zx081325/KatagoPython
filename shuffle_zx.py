#!/usr/bin/python3
import sys
import os
import argparse
import time
import zipfile
import shutil
import psutil
import json
import hashlib
import gc

import multiprocessing

import numpy as np


def assert_keys(npz, include_meta):
    keys = [
        "binaryInputNCHWPacked",
        "globalInputNC",
        "policyTargetsNCMove",
        "globalTargetsNC",
        "scoreDistrN",
        "valueTargetsNCHW",
    ]
    if include_meta:
        keys.append("metadataInputNC")
    assert (set(npz.keys()) == set(keys))


# 检查是否是临时文件
def is_temp_npz_like(filename):
    return "_" in filename


# 重新排列并取前n个值
def joint_shuffle_take_first_n(n, arrs):
    for arr in arrs:
        assert (len(arr) == len(arrs[0]))
    perm = np.random.permutation(len(arrs[0]))
    perm = perm[:n]
    shuffled_arrs = []
    for arr in arrs:
        shuffled_arrs.append(arr[perm])
    return shuffled_arrs


def memusage_mb():
    return psutil.Process(os.getpid()).memory_info().rss // 1048576


def shardify(input_idx, input_file_group, num_out_files, out_tmp_dirs, keep_prob, include_meta):
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for _ in range(4)])

    assert (len(input_file_group) > 0)
    num_files_not_found = 0

    binaryInputNCHWPackedList = []
    globalInputNCList = []
    policyTargetsNCMoveList = []
    globalTargetsNCList = []
    scoreDistrNList = []
    valueTargetsNCHWList = []
    metadataInputNCList = []

    for input_file in input_file_group:
        try:
            with np.load(input_file) as npz:
                assert_keys(npz, include_meta)
                binaryInputNCHWPackedList.append(npz["binaryInputNCHWPacked"])
                globalInputNCList.append(npz["globalInputNC"])
                policyTargetsNCMoveList.append(npz["policyTargetsNCMove"])
                globalTargetsNCList.append(npz["globalTargetsNC"])
                scoreDistrNList.append(npz["scoreDistrN"])
                valueTargetsNCHWList.append(npz["valueTargetsNCHW"])
                metadataInputNCList.append(npz["metadataInputNC"] if include_meta else None)
        except FileNotFoundError:
            num_files_not_found += 1
            print("WARNING: file not found by shardify: ", input_file)
            pass
    if len(binaryInputNCHWPackedList) <= 0:
        return num_files_not_found  # Early quit since we don't know shapes

    binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackedList, axis=0)
    globalInputNC = np.concatenate(globalInputNCList, axis=0)
    policyTargetsNCMove = np.concatenate(policyTargetsNCMoveList, axis=0)
    globalTargetsNC = np.concatenate(globalTargetsNCList, axis=0)
    scoreDistrN = np.concatenate(scoreDistrNList, axis=0)
    valueTargetsNCHW = np.concatenate(valueTargetsNCHWList, axis=0)
    metadataInputNC = np.concatenate(metadataInputNCList, axis=0) if include_meta else None

    # 数据校验
    num_rows_to_keep = binaryInputNCHWPacked.shape[0]
    assert (globalInputNC.shape[0] == num_rows_to_keep)
    assert (policyTargetsNCMove.shape[0] == num_rows_to_keep)
    assert (globalTargetsNC.shape[0] == num_rows_to_keep)
    assert (scoreDistrN.shape[0] == num_rows_to_keep)
    assert (valueTargetsNCHW.shape[0] == num_rows_to_keep)
    assert (metadataInputNC.shape[0] == num_rows_to_keep if include_meta else True)

    if keep_prob < 1.0:
        num_rows_to_keep = int(round(num_rows_to_keep * keep_prob))

    if include_meta:
        [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW,
         metadataInputNC] = (
            joint_shuffle_take_first_n(
                num_rows_to_keep,
                [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN,
                 valueTargetsNCHW, metadataInputNC]
            )
        )
    else:
        [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW] = (
            joint_shuffle_take_first_n(
                num_rows_to_keep,
                [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN,
                 valueTargetsNCHW]
            )
        )

    assert (binaryInputNCHWPacked.shape[0] == num_rows_to_keep)
    assert (globalInputNC.shape[0] == num_rows_to_keep)
    assert (policyTargetsNCMove.shape[0] == num_rows_to_keep)
    assert (globalTargetsNC.shape[0] == num_rows_to_keep)
    assert (scoreDistrN.shape[0] == num_rows_to_keep)
    assert (valueTargetsNCHW.shape[0] == num_rows_to_keep)
    assert (metadataInputNC.shape[0] == num_rows_to_keep if include_meta else True)

    # 保存了一个长度为num_rows_to_keep的数组，每个元素是一个随机的整数[0, num_out_files - 1]，表示对应的行被分配到哪个文件
    rand_assts = np.random.randint(num_out_files, size=[num_rows_to_keep])
    # 每个文件被分配的行数
    counts = np.bincount(rand_assts, minlength=num_out_files)
    # 前i个文件被分配的总行数
    countsums = np.cumsum(counts)
    assert (countsums[len(countsums) - 1] == num_rows_to_keep)

    for out_idx in range(num_out_files):
        start = countsums[out_idx] - counts[out_idx]
        stop = countsums[out_idx]
        if include_meta:
            np.savez_compressed(
                os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
                binaryInputNCHWPacked=binaryInputNCHWPacked[start:stop],
                globalInputNC=globalInputNC[start:stop],
                policyTargetsNCMove=policyTargetsNCMove[start:stop],
                globalTargetsNC=globalTargetsNC[start:stop],
                scoreDistrN=scoreDistrN[start:stop],
                valueTargetsNCHW=valueTargetsNCHW[start:stop],
                metadataInputNC=metadataInputNC[start:stop],
            )
        else:
            np.savez_compressed(
                os.path.join(out_tmp_dirs[out_idx], str(input_idx) + ".npz"),
                binaryInputNCHWPacked=binaryInputNCHWPacked[start:stop],
                globalInputNC=globalInputNC[start:stop],
                policyTargetsNCMove=policyTargetsNCMove[start:stop],
                globalTargetsNC=globalTargetsNC[start:stop],
                scoreDistrN=scoreDistrN[start:stop],
                valueTargetsNCHW=valueTargetsNCHW[start:stop],
            )
    return num_files_not_found


# 合并分片
def merge_shards(filename, num_shards_to_merge, out_tmp_dir, batch_size, ensure_batch_multiple, include_meta):
    # 使用了一个包含 5 个随机整数的列表作为种子，增强随机性的复杂性和不可预测性。
    np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(5)])

    binaryInputNCHWPackeds = []
    globalInputNCs = []
    policyTargetsNCMoves = []
    globalTargetsNCs = []
    scoreDistrNs = []
    valueTargetsNCHWs = []
    metadataInputNCs = []

    for input_idx in range(num_shards_to_merge):
        shard_filename = os.path.join(out_tmp_dir, str(input_idx) + ".npz")
        try:
            with np.load(shard_filename) as npz:
                assert_keys(npz, include_meta)

                binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
                globalInputNC = npz["globalInputNC"]
                policyTargetsNCMove = npz["policyTargetsNCMove"]
                globalTargetsNC = npz["globalTargetsNC"]
                scoreDistrN = npz["scoreDistrN"]
                valueTargetsNCHW = npz["valueTargetsNCHW"]
                metadataInputNC = npz["metadataInputNC"] if include_meta else None

                binaryInputNCHWPackeds.append(binaryInputNCHWPacked)
                globalInputNCs.append(globalInputNC)
                policyTargetsNCMoves.append(policyTargetsNCMove)
                globalTargetsNCs.append(globalTargetsNC)
                scoreDistrNs.append(scoreDistrN)
                valueTargetsNCHWs.append(valueTargetsNCHW)
                metadataInputNCs.append(metadataInputNC)
        except FileNotFoundError:
            print("WARNING: Empty shard in merge_shards for shard :", input_idx, filename)

    if len(binaryInputNCHWPackeds) <= 0:
        print("WARNING: empty merge file: ", filename)
        return 0

    ###
    # WARNING - if adding anything here, also add it to joint_shuffle below!
    ###
    binaryInputNCHWPacked = np.concatenate(binaryInputNCHWPackeds)
    globalInputNC = np.concatenate(globalInputNCs)
    policyTargetsNCMove = np.concatenate(policyTargetsNCMoves)
    globalTargetsNC = np.concatenate(globalTargetsNCs)
    scoreDistrN = np.concatenate(scoreDistrNs)
    valueTargetsNCHW = np.concatenate(valueTargetsNCHWs)
    metadataInputNC = np.concatenate(metadataInputNCs) if include_meta else None

    num_rows = binaryInputNCHWPacked.shape[0]
    assert (globalInputNC.shape[0] == num_rows)
    assert (policyTargetsNCMove.shape[0] == num_rows)
    assert (globalTargetsNC.shape[0] == num_rows)
    assert (scoreDistrN.shape[0] == num_rows)
    assert (valueTargetsNCHW.shape[0] == num_rows)
    assert (metadataInputNC.shape[0] == num_rows if include_meta else True)

    if include_meta:
        [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW,
         metadataInputNC] = (
            joint_shuffle_take_first_n(
                num_rows,
                [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN,
                 valueTargetsNCHW, metadataInputNC],
            )
        )
    else:
        [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN, valueTargetsNCHW] = (
            joint_shuffle_take_first_n(
                num_rows,
                [binaryInputNCHWPacked, globalInputNC, policyTargetsNCMove, globalTargetsNC, scoreDistrN,
                 valueTargetsNCHW],
            )
        )

    assert (binaryInputNCHWPacked.shape[0] == num_rows)
    assert (globalInputNC.shape[0] == num_rows)
    assert (policyTargetsNCMove.shape[0] == num_rows)
    assert (globalTargetsNC.shape[0] == num_rows)
    assert (scoreDistrN.shape[0] == num_rows)
    assert (valueTargetsNCHW.shape[0] == num_rows)
    assert (metadataInputNC.shape[0] == num_rows if include_meta else True)

    # Just truncate and lose the batch at the end, it's fine
    num_batches = (num_rows // (batch_size * ensure_batch_multiple)) * ensure_batch_multiple
    start = 0
    stop = num_batches * batch_size
    if include_meta:
        np.savez_compressed(
            filename,
            binaryInputNCHWPacked=binaryInputNCHWPacked[start:stop],
            globalInputNC=globalInputNC[start:stop],
            policyTargetsNCMove=policyTargetsNCMove[start:stop],
            globalTargetsNC=globalTargetsNC[start:stop],
            scoreDistrN=scoreDistrN[start:stop],
            valueTargetsNCHW=valueTargetsNCHW[start:stop],
            metadataInputNC=metadataInputNC[start:stop],
        )
    else:
        np.savez_compressed(
            filename,
            binaryInputNCHWPacked=binaryInputNCHWPacked[start:stop],
            globalInputNC=globalInputNC[start:stop],
            policyTargetsNCMove=policyTargetsNCMove[start:stop],
            globalTargetsNC=globalTargetsNC[start:stop],
            scoreDistrN=scoreDistrN[start:stop],
            valueTargetsNCHW=valueTargetsNCHW[start:stop],
        )
    jsonfilename = os.path.splitext(filename)[0] + ".json"
    with open(jsonfilename, "w") as f:
        json.dump({"num_rows": num_rows, "num_batches": num_batches}, f)

    return num_batches * batch_size


# 获取npz文件头信息
def get_numpy_npz_headers(filename):
    with zipfile.ZipFile(filename) as z:
        wasbad = False
        npzheaders = {}
        # 遍历压缩文件中所有文件名
        for subfilename in z.namelist():
            npyfile = z.open(subfilename)
            try:
                version = np.lib.format.read_magic(npyfile)
            except ValueError:
                wasbad = True
                print("WARNING: 文件有问题，跳过: %s (坏的数组 %s)" % (filename, subfilename))
            else:
                (shape, is_fortran, dtype) = np.lib.format._read_array_header(npyfile, version)
                npzheaders[subfilename] = (shape, is_fortran, dtype)
        if wasbad:
            return None
        return npzheaders


# 计算文件行数
def compute_num_rows(filename):
    try:
        npheaders = get_numpy_npz_headers(filename)
    except PermissionError:
        print("WARNING: 无权限读取文件: ", filename)
        return (filename, None)
    except zipfile.BadZipFile:
        print("WARNING: 文件是损坏的 zip 文件: ", filename)
        return (filename, None)
    if npheaders is None or len(npheaders) <= 0:
        print("WARNING: 文件的 npz 头信息有误: ", filename)
        return (filename, None)

    if "binaryInputNCHWPacked" in npheaders:
        (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
    else:
        (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked.npy"]
    num_rows = shape[0]
    return (filename, num_rows)


class TimeStuff(object):

    def __init__(self, taskstr):
        self.taskstr = taskstr

    def __enter__(self):
        print("Beginning: %s" % self.taskstr, flush=True)
        self.t0 = time.time()

    def __exit__(self, exception_type, exception_val, trace):
        self.t1 = time.time()
        print("Finished: %s in %s seconds" % (self.taskstr, str(self.t1 - self.t0)), flush=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter, description="""
        打乱数据文件！

        这个打乱脚本是为持续自对弈训练设计的。它会在提供的数据中打乱最近的一个窗口的数据，并根据运行至今的总数据量动态选择窗口大小，假设提供的目录包含了整个运行过程中的所有数据。如果你实际上并没有所有的数据，
        例如你已归档或删除了旧数据，或者希望按照有更多数据的情况来计算窗口大小，可以使用 -add-to-data-rows 选项。

        窗口大小是基于运行中数据行数 N 的幂律：
          WINDOWSIZE(N) = (N^EXPONENT - MIN_ROWS^EXPONENT) / (EXPONENT * MIN_ROWS^(EXPONENT-1)) * INITIAL_WINDOW_PER_ROW + MIN_ROWS

        需要传递的参数：
          -taper-window-exponent EXPONENT \\
          -expand-window-per-row INITIAL_WINDOW_PER_ROW \\
          -min-rows MIN_ROWS  (默认值为 25 万)

        这看起来可能有点复杂，但基本上它只是一个简单的幂律 N^EXPONENT，并通过平移和缩放使其满足以下条件：
        WINDOWSIZE(MIN_ROWS) = MIN_ROWS
        (dWINDOWSIZE/dN)(MIN_ROWS) = INITIAL_WINDOW_PER_ROW

        类似于 KataGo 主要运行中使用的合理参数可以是：
          -taper-window-exponent 0.65 或 0.675 \\
          -expand-window-per-row 0.4 \\
          -min-rows 250000（默认值）

        如果你希望控制幂律的“规模”不同于最小行数，你可以指定 -taper-window-scale 选项。
        此外，还有一个稍显不太优雅的处理方式，用于限制随机行的数量（由不使用神经网络的随机走子生成的行），因为在运行初期，由于未使用 GPU，随机行生成速度可能非常快，从而导致运行中的数据过量。

        另外，并不是整个打乱的窗口都会输出，只有随机打乱的 2000 万行会被保留。你可以使用 -keep-target-rows 来调整这个值。此脚本的意图是，随着新数据的进入，会重复运行该脚本，以便在 train.py 需要超过 2000 万行之前，数据已被重新打乱，并选择了新的随机 2000 万行。

        如果你不进行持续的自对弈训练，而只是想打乱整个数据集（而不仅仅是一个窗口）并且想输出全部数据（而不仅仅是 2000 万行），那么可以使用以下参数：
          -taper-window-exponent 1.0 \\
          -expand-window-per-row 1.0 \\
          -keep-target-rows SOME_VERY_LARGE_NUMBER

        如果你正在进行持续的自对弈训练，但希望使用固定窗口大小，则可以使用以下参数：
          -min-rows 你期望的大小 \\
          -taper-window-exponent 1.0 \\
          -expand-window-per-row 0.0
        """)

    parser.add_argument('dirs', metavar='DIR', nargs='*', default=['data\\selfplay'], help='训练数据文件的目录')

    required_args = parser.add_argument_group('必选参数')
    optional_args = parser.add_argument_group('可选参数')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='显示此帮助信息并退出'
    )
    optional_args.add_argument('-min-rows', type=int, required=False, default=250000,
                               help='使用的最少训练行数，默认 25 万')
    optional_args.add_argument('-max-rows', type=int, required=False, help='使用的最多训练行数，默认为无限制')
    optional_args.add_argument('-keep-target-rows', type=int, required=False, default=20000000,
                               help='最终数据集中保留的目标行数，默认 2000 万')
    required_args.add_argument('-expand-window-per-row', type=float, required=True,
                               help='超过最小行数后，每行随机数据将使窗口扩展这么多')
    required_args.add_argument('-taper-window-exponent', type=float, required=True,
                               help='使窗口大小渐近增长为数据行数的这个次方')
    optional_args.add_argument('-taper-window-scale', type=float, required=False,
                               help='幂律适用的比例，默认为 -min-rows')
    optional_args.add_argument('-add-to-data-rows', type=float, required=False, default=0,
                               help='计算窗口大小时，将数据行数视为比实际数量多/少这么多')
    optional_args.add_argument('-summary-file', required=False, help='用于目录内容的概要 json 文件')
    required_args.add_argument('-out-dir', required=True, help='输出训练文件的目录')
    required_args.add_argument('-out-tmp-dir', required=True, help='用作临时空间的目录')
    optional_args.add_argument('-approx-rows-per-out-file', type=int, required=False, default=70000,
                               help='每个输出文件的行数，默认为 7 万')
    required_args.add_argument('-num-processes', type=int, required=True, help='用于并行打乱的多进程数量')
    required_args.add_argument('-batch-size', type=int, required=True, help='写入训练样本的批量大小')
    optional_args.add_argument('-ensure-batch-multiple', type=int, required=False, default=1,
                               help='确保每个文件的行数是此批量大小的倍数')
    optional_args.add_argument('-worker-group-size', type=int, required=False, default=80000,
                               help='内部目标是每个并行分片工作者处理这么多行（不影响合并）')
    optional_args.add_argument('-exclude', required=False, default=None,
                               help='包含要忽略的 npz 文件的文本文件，每行一个文件')
    optional_args.add_argument('-exclude-prefix', required=False, help='将前缀拼接到排除文件中的行，生成完整的文件路径')
    optional_args.add_argument('-exclude-basename', required=False, action="store_true",
                               help='如果匹配到的基名一致，则视为排除')
    optional_args.add_argument('-only-include-md5-path-prop-lbound', type=float, required=False,
                               help='在分片前，仅包括哈希到浮点数 >= 此值的文件路径')
    optional_args.add_argument('-only-include-md5-path-prop-ubound', type=float, required=False,
                               help='在分片前，仅包括哈希到浮点数 < 此值的文件路径')
    optional_args.add_argument('-include-meta', action="store_true", required=False, help='包含 sgf 元数据输入')

    args = parser.parse_args()
    # 数据目录列表
    dirs = args.dirs
    # 最小训练行数, 默认25w
    min_rows = args.min_rows
    # 最大选择行数，非训练行数
    max_rows = args.max_rows
    # 最终数据集中保留的目标行数，默认2000万
    keep_target_rows = args.keep_target_rows
    # 超过最小行数后，每行随机数据将使窗口扩展这么多
    expand_window_per_row = args.expand_window_per_row
    # 使窗口大小渐近增长为数据行数的这个次方
    taper_window_exponent = args.taper_window_exponent
    taper_window_scale = args.taper_window_scale
    add_to_data_rows = args.add_to_data_rows
    # 摘要文件
    summary_file = args.summary_file
    # 输出目录
    out_dir = args.out_dir
    # 输出临时目录
    out_tmp_dir = args.out_tmp_dir
    # 每个输出文件的行数
    approx_rows_per_out_file = args.approx_rows_per_out_file
    # 进程数
    num_processes = args.num_processes
    # 批次大小
    batch_size = args.batch_size
    # 确保每个文件的行数是批量大小的这么多倍
    ensure_batch_multiple = args.ensure_batch_multiple
    # 每个并行分片处理的行数，默认80000
    worker_group_size = args.worker_group_size
    # 排除的文件
    exclude = args.exclude
    # 排除文件的前缀
    exclude_prefix = args.exclude_prefix
    # 是否去掉路径比较
    exclude_basename = args.exclude_basename
    # 文件转为md5并映射成[0,1]所取下界
    only_include_md5_path_prop_lbound = args.only_include_md5_path_prop_lbound
    # 文件转为md5并映射成[0,1]所取上界
    only_include_md5_path_prop_ubound = args.only_include_md5_path_prop_ubound
    # 是否包含sgf元数据
    include_meta = args.include_meta

    summary_data_by_dirpath = {}
    if summary_file is not None:
        with TimeStuff("加载 " + summary_file):
            # 尝试多次加载文件，以确保在文件被nfs替换时具有鲁棒性
            for i in range(10):
                success = False
                try:
                    with open(summary_file) as fp:
                        summary_data_by_dirpath = json.load(fp)
                        success = True
                except OSError:
                    success = False
                except ValueError:
                    success = False
                if success:
                    break
                time.sleep(1)
            if not success:
                raise RuntimeError("无法加载摘要文件")

    exclude_set = set()
    if exclude is not None:
        with TimeStuff("加载 " + exclude):
            # 尝试多次加载文件，以确保在文件被nfs替换时具有鲁棒性
            for i in range(10):
                success = False
                try:
                    with open(exclude, "r") as exclude_in:
                        excludes = exclude_in.readlines()
                        excludes = [x.strip() for x in excludes]
                        excludes = [x for x in excludes if len(x) > 0]
                        excludes = [exclude_prefix + x for x in excludes]
                        exclude_set = set(excludes)
                        success = True
                except OSError:
                    success = False
                except ValueError:
                    success = False
                if success:
                    break
                time.sleep(1)
            if not success:
                raise RuntimeError("无法加载排除文件")

    # 取去掉路径后的文件名部分
    if exclude_basename:
        basenames = [os.path.basename(path) for path in exclude_set]
        exclude_set.update(basenames)

    all_files = []
    files_with_unknown_num_rows = []
    excluded_count = 0
    excluded_due_to_excludes_count = 0
    tempfilelike_count = 0
    with TimeStuff("Finding files"):
        for d in dirs:
            # 遍历每个目录
            for (path, dirnames, filenames) in os.walk(d, followlinks=True):
                i = 0
                while i < len(dirnames):
                    dirname = dirnames[i]
                    # 获取当前目录的摘要数据
                    summary_data = summary_data_by_dirpath.get(os.path.abspath(os.path.join(path, dirname)))
                    if summary_data is not None:
                        filename_mtime_num_rowss = summary_data["filename_mtime_num_rowss"]
                        # 如果目录有摘要数据，删除该目录
                        del dirnames[i]
                        i -= 1
                        # 处理摘要数据中的每个文件
                        for (filename, mtime, num_rows) in filename_mtime_num_rowss:
                            if is_temp_npz_like(filename):
                                # 检查是否为临时文件，如果是，则排除
                                # print("WARNING: file looks like a temp file, treating as exclude: ", os.path.join(path, dirname, filename))
                                excluded_count += 1
                                tempfilelike_count += 1
                                continue
                            if exclude_basename and os.path.basename(filename) in exclude_set:
                                # 如果文件名在排除列表中，排除该文件
                                excluded_count += 1
                                excluded_due_to_excludes_count += 1
                                continue
                            filename = os.path.join(path, dirname, filename)
                            if not exclude_basename and filename in exclude_set:
                                # 如果文件路径在排除列表中，排除该文件
                                excluded_count += 1
                                excluded_due_to_excludes_count += 1
                                continue
                            if num_rows is None:
                                # 如果行数未知，跳过该文件
                                print("WARNING: Skipping bad rowless file, treating as exclude: ", filename)
                                excluded_count += 1
                                continue
                            # 将文件添加到结果列表中
                            all_files.append((filename, mtime, num_rows))
                    i += 1

                filtered_filenames = []
                # 过滤出后缀为".npz"的文件
                for filename in filenames:
                    if not filename.endswith(".npz"):
                        continue
                    if is_temp_npz_like(filename):
                        # 检查是否为临时文件，如果是，则排除
                        excluded_count += 1
                        tempfilelike_count += 1
                        continue
                    if exclude_basename and os.path.basename(filename) in exclude_set:
                        # 如果文件名在排除列表中，排除该文件
                        excluded_count += 1
                        excluded_due_to_excludes_count += 1
                        continue
                    # 生成完整的文件路径
                    filename = os.path.join(path, filename)
                    if not exclude_basename and filename in exclude_set:
                        # 如果文件路径在排除列表中，排除该文件
                        excluded_count += 1
                        excluded_due_to_excludes_count += 1
                        continue
                    filtered_filenames.append(filename)
                filenames = filtered_filenames

                # 将未知行数的文件加入列表
                files_with_unknown_num_rows.extend(filenames)
                # 获取每个文件的修改时间并加入结果列表
                filenames = [(filename, os.path.getmtime(filename)) for filename in filenames]
                all_files.extend(filenames)

    print("文件总数: %d" % len(all_files), flush=True)
    print("行数未知的文件总数: %d" % len(files_with_unknown_num_rows), flush=True)
    print("排除的文件数: %d" % excluded_count, flush=True)
    print("因为看起来像临时文件而被排除的文件数: %d" % tempfilelike_count, flush=True)
    print("因为命令行排除文件而被排除的文件数: %d" % excluded_due_to_excludes_count, flush=True)

    print("执行垃圾回收", flush=True)
    del summary_data_by_dirpath
    gc.collect()

    # 按照修改时间升序排序（猜测可能是先处理历史文件留一点时间给最新的文件加载完成）
    with TimeStuff("文件排序"):
        all_files.sort(key=(lambda x: x[1]), reverse=False)

    # 等待几秒钟，以减少文件系统竞争的可能性
    time.sleep(3)

    with TimeStuff("计算文件行数"):
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(compute_num_rows, files_with_unknown_num_rows)
            results = dict(results)
            for i in range(len(all_files)):
                info = all_files[i]
                if len(info) < 3:
                    num_rows = results[info[0]]
                    all_files[i] = (info[0], info[1], num_rows)

    num_rows_total = 0  # 数据总行数
    num_random_rows_capped = 0  # 随机数据行数（未训练过的神经网络），最多保留 min_rows 行
    num_postrandom_rows = 0  # 非随机数据行数

    # 我们从幂律窗口尾部开始的偏移量是多少？例如，我们需要多少个非随机数据行才能使窗口大小增长一个因子 2^(taper_window_exponent)？
    # 目前，我们将其设置为 min_rows
    if taper_window_scale is not None:
        window_taper_offset = taper_window_scale
    else:
        window_taper_offset = min_rows


    def num_usable_rows():
        global num_random_rows_capped
        global num_postrandom_rows
        return num_random_rows_capped + num_postrandom_rows


    # 计算期望选择的行数, 这里做了个非线性转换，总行数越多期望行数增长越慢
    def num_desired_rows():
        power_law_x = num_usable_rows() - min_rows + window_taper_offset + add_to_data_rows
        unscaled_power_law = (power_law_x ** taper_window_exponent) - (window_taper_offset ** taper_window_exponent)
        scaled_power_law = unscaled_power_law / (
                taper_window_exponent * (window_taper_offset ** (taper_window_exponent - 1)))
        return int(scaled_power_law * expand_window_per_row + min_rows)


    with TimeStuff("处理找到的文件"):
        for (filename, mtime, num_rows) in all_files:
            if num_rows is None:
                print("警告: 跳过无效的文件: ", filename)
                continue
            if num_rows <= 0:
                continue
            num_rows_total += num_rows
            # 是否是随机数据行，随机数据行最多使用min_rows条
            if "random/tdata/" not in filename and "random\\tdata\\" not in filename:
                num_postrandom_rows += num_rows
            else:
                num_random_rows_capped = min(num_random_rows_capped + num_rows, min_rows)

    if os.path.exists(out_dir):
        raise Exception(out_dir + " 已经存在，退出程序")
    os.mkdir(out_dir)

    if num_rows_total <= 0:
        print("没有数据，退出程序")
        sys.exit(0)

    # 如果行数不足，则退出程序
    if num_rows_total < min_rows:
        print("行数不足，仅有 %d 行（少于 %d 行）" % (num_rows_total, min_rows))
        sys.exit(0)

    print("总行数: %d (%d 可用)" % (num_rows_total, num_usable_rows()), flush=True)

    # 逆序，按照日期从近到远排序
    all_files.reverse()

    # 计算期望选择的行数
    desired_num_rows = num_desired_rows()
    desired_num_rows = max(desired_num_rows, min_rows)
    desired_num_rows = min(desired_num_rows, max_rows) if max_rows is not None else desired_num_rows
    print("期望的行数: %d / %d" % (desired_num_rows, num_rows_total), flush=True)

    desired_input_files = []
    min_start_row = num_rows_total
    max_end_row = num_rows_total
    num_rows_used = 0
    print_stride = 1 + len(all_files) // 80
    end_row = num_rows_total
    # 数据从最新的文件开始往前取，直到满足需要的数据行数
    with TimeStuff("计算所需的行数"):
        for i in range(len(all_files)):
            (filename, mtime, num_rows) = all_files[i]

            if num_rows is not None and num_rows > 0:
                desired_input_files.append((filename, num_rows))
                start_row = end_row - num_rows
                min_start_row = min(start_row, min_start_row)
                num_rows_used += num_rows
                end_row -= num_rows
            else:
                start_row = end_row

            if i % print_stride == 0 or num_rows_used >= desired_num_rows:
                print("使用中: %s (%d-%d) (%d/%d 需要的行数)" % (
                    filename, start_row, end_row, num_rows_used, desired_num_rows), flush=True)
            if num_rows_used >= desired_num_rows:
                break

    print("最后使用: (%d-%d) (%d/%d 需要的行数)" % (min_start_row, max_end_row, num_rows_used, desired_num_rows),
          flush=True)

    print("执行垃圾回收", flush=True)
    del all_files
    gc.collect()

    np.random.seed()
    # 打乱文件顺序
    np.random.shuffle(desired_input_files)

    approx_rows_to_keep = num_rows_used
    if keep_target_rows is not None:
        approx_rows_to_keep = min(approx_rows_to_keep, keep_target_rows)
    keep_prob = approx_rows_to_keep / num_rows_used

    num_out_files = int(round(approx_rows_to_keep / approx_rows_per_out_file))
    num_out_files = max(num_out_files, 1)

    out_files = [os.path.join(out_dir, "data%d.npz" % i) for i in range(num_out_files)]

    out_tmp_dirs = [os.path.join(out_tmp_dir, "tmp.shuf%d" % i) for i in range(num_out_files)]
    print("正在写入 %d 个输出文件，保留了 %d 行 / 需要 %d 行" % (num_out_files, approx_rows_to_keep, desired_num_rows),
          flush=True)


    def clean_tmp_dirs():
        for tmp_dir in out_tmp_dirs:
            if os.path.exists(tmp_dir):
                print("清理临时目录: " + tmp_dir)
                shutil.rmtree(tmp_dir)


    clean_tmp_dirs()
    for tmp_dir in out_tmp_dirs:
        os.mkdir(tmp_dir)

    num_rows_in_desired_files = 0  # 初始化符合条件的文件总行数为0

    # 根据md5值筛选出（only_include_md5_path_prop_ubound-only_include_md5_path_prop_lbound）比例的文件，用来区分测试集和验证集
    if only_include_md5_path_prop_lbound is not None or only_include_md5_path_prop_ubound is not None:
        new_desired_input_files = []
        for (input_file, num_rows_in_file) in desired_input_files:
            input_file_base = os.path.basename(input_file)
            # 计算文件名的MD5哈希值前13位，转化为浮点数
            hashfloat = int("0x" + hashlib.md5(str(input_file_base).encode('utf-8')).hexdigest()[:13], 16) / 2 ** 52
            ok = True  # 假设文件符合条件
            if only_include_md5_path_prop_lbound is not None and hashfloat < only_include_md5_path_prop_lbound:
                ok = False
            if only_include_md5_path_prop_ubound is not None and hashfloat >= only_include_md5_path_prop_ubound:
                ok = False
            if ok:
                new_desired_input_files.append((input_file, num_rows_in_file))
                num_rows_in_desired_files += num_rows_in_file
        print("根据md5筛选出 %d/%d 个文件" % (len(new_desired_input_files), len(desired_input_files)))
        desired_input_files = new_desired_input_files  # 更新符合条件的文件列表
    else:
        # 如果没有MD5限制条件，直接累加所有文件的行数
        for (input_file, num_rows_in_file) in desired_input_files:
            num_rows_in_desired_files += num_rows_in_file

    # 如果没有符合条件的文件，退出程序
    if len(desired_input_files) <= 0:
        print("筛选后没有文件")
        sys.exit(0)
    # 如果符合条件的文件没有行数，退出程序
    if num_rows_in_desired_files <= 0:
        print("符合条件的文件没有行数")
        sys.exit(0)

    desired_input_file_groups = []
    group_size_so_far = 0
    group_so_far = []
    for (input_file, num_rows_in_file) in desired_input_files:
        if num_rows_in_file <= 0:
            continue
        group_so_far.append(input_file)
        group_size_so_far += num_rows_in_file
        if group_size_so_far >= worker_group_size:
            desired_input_file_groups.append(group_so_far)
            group_so_far = []
            group_size_so_far = 0
    if group_size_so_far > 0:
        desired_input_file_groups.append(group_so_far)
        group_so_far = []
        group_size_so_far = 0
    print("将 %d 个输入文件分组成 %d 个分片组" % (len(desired_input_files), len(desired_input_file_groups)), flush=True)

    # 使用多进程池处理
    with multiprocessing.Pool(num_processes) as pool:
        with TimeStuff("分片处理"):
            shard_results = pool.starmap(shardify, [
                (input_idx, desired_input_file_groups[input_idx], num_out_files, out_tmp_dirs, keep_prob, include_meta)
                for input_idx in range(len(desired_input_file_groups))
            ])

        with TimeStuff("合并分片"):
            num_shards_to_merge = len(desired_input_file_groups)
            merge_results = pool.starmap(merge_shards, [
                (
                    out_files[idx], num_shards_to_merge, out_tmp_dirs[idx], batch_size, ensure_batch_multiple,
                    include_meta)
                for idx in range(len(out_files))
            ])
        print("每个输出文件的行数如下:", flush=True)
        print(list(zip(out_files, merge_results)), flush=True)
        sys.stdout.flush()

    clean_tmp_dirs()

    dump_value = {
        "range": (min_start_row, max_end_row)
    }

    with open(out_dir + ".json", 'w') as f:
        json.dump(dump_value, f)

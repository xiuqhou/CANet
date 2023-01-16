from argparse import ArgumentParser

import mmcv

from mmdet import datasets
from mmdet.core import eval_map, eval_recalls


def voc_eval(result_file, dataset, iou_thr=0.5, nproc=4, mAP=True):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    if mAP:
        eval_map(
            det_results,
            annotations,
            scale_ranges=None,
            iou_thr=iou_thr, # 阈值为多少认为是匹配上，mAP50~iou=0.5
            dataset=dataset_name,
            logger='print',
            nproc=nproc)
    else:
        eval_recalls(annotations, det_results, iou_thr=iou_thr, logger='print')


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument( # 用来评估recalls
        '--mAP',
        default=True,
        help='Eval mAP or recalls')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        help='Processes to be used for computing mAP')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr, args.nproc, args.mAP)


if __name__ == '__main__':
    main()
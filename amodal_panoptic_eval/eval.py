#!/usr/bin/python
#
# The evaluation script for amodal panoptic segmentation (https://arxiv.org/abs/2202.11542).
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
# Test set evaluation assumes prediction use 'id' and not 'trainId'
# for categories, i.e. 'person' id is 24.
# The script computes the Amodal Panoptic Quality (APQ) metric and the Amodal Parsing Coverage (APC) metric.

from __future__ import print_function, absolute_import, division, unicode_literals
import os
import sys
import argparse
import json
import time
import multiprocessing
import numpy as np
from collections import defaultdict
from scipy import optimize


import cv2

import pycocotools.mask as mask_utils
from .labels.asd_labels import id2label as asd_id2label

import tqdm

OFFSET = 1000
VOID = 0


def print_error(message):
    print("ERROR: " + str(message))
    sys.exit(-1)


class APQStatCat:
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.occ_iou = 0.0
        self.occ_tp = 0
        self.occ_fp = 0
        self.occ_fn = 0

    def __iadd__(self, apq_stat_cat):
        self.iou += apq_stat_cat.iou
        self.tp += apq_stat_cat.tp
        self.fp += apq_stat_cat.fp
        self.fn += apq_stat_cat.fn
        self.occ_iou += apq_stat_cat.occ_iou
        self.occ_tp += apq_stat_cat.occ_tp
        self.occ_fp += apq_stat_cat.occ_fp
        self.occ_fn += apq_stat_cat.occ_fn
        return self


class APQStat:
    def __init__(self):
        self.apq_per_cat = defaultdict(APQStatCat)

    def __getitem__(self, i):
        return self.apq_per_cat[i]

    def __iadd__(self, apq_stat):
        for label, apq_stat_cat in apq_stat.apq_per_cat.items():
            self.apq_per_cat[label] += apq_stat_cat
        return self

    def apq_average(self, categories):
        stuff_per_class_results = {}
        things_per_class_results = {}
        for label_type, label_info in categories.items():
            if label_type == "stuff":
                for label_id, label in label_info.items():
                    total_segments = (
                        self.apq_per_cat[label_id].tp + self.apq_per_cat[label_id].fn
                    )
                    if total_segments > 0:
                        apq_sc = self.apq_per_cat[label_id].iou / total_segments
                    else:
                        apq_sc = 0.0
                    stuff_per_class_results[label.name] = {"apq_sc": apq_sc}
            else:
                for label_id, label in label_info.items():
                    total_visible_segments = (
                        self.apq_per_cat[label_id].tp + self.apq_per_cat[label_id].fn
                    )
                    total_occ_segments = (
                        self.apq_per_cat[label_id].occ_tp
                        + self.apq_per_cat[label_id].occ_fn
                    )
                    if total_visible_segments > 0:
                        apq_vtc = self.apq_per_cat[label_id].iou / (
                            total_visible_segments + self.apq_per_cat[label_id].fp
                        )
                    else:
                        apq_vtc = 0.0
                    if total_occ_segments > 0:
                        apq_otc = self.apq_per_cat[label_id].occ_iou / (
                            total_occ_segments + self.apq_per_cat[label_id].occ_fp
                        )
                    else:
                        apq_otc = 0.0
                    if total_visible_segments + total_occ_segments > 0:
                        apq_tc = (
                            self.apq_per_cat[label_id].iou
                            + self.apq_per_cat[label_id].occ_iou
                        ) / (
                            total_visible_segments
                            + total_occ_segments
                            + self.apq_per_cat[label_id].fp
                            + self.apq_per_cat[label_id].occ_fp
                        )
                    else:
                        apq_tc = 0.0
                    things_per_class_results[label.name] = {
                        "apq_vtc": apq_vtc,
                        "apq_otc": apq_otc,
                        "apq_tc": apq_tc,
                    }

        n_stuff, n_things = len(stuff_per_class_results), len(things_per_class_results)
        apq_stuff = (
            np.sum(
                np.array(
                    [
                        stuff_per_class_results[label]["apq_sc"]
                        for label in stuff_per_class_results
                    ]
                )
            )
            / n_stuff
        )
        apq_things = (
            np.sum(
                np.array(
                    [
                        things_per_class_results[label]["apq_tc"]
                        for label in things_per_class_results
                    ]
                )
            )
            / n_things
        )
        apq_vis = (
            np.sum(
                np.array(
                    [
                        things_per_class_results[label]["apq_vtc"]
                        for label in things_per_class_results
                    ]
                )
            )
            / n_things
        )
        apq_occ = (
            np.sum(
                np.array(
                    [
                        things_per_class_results[label]["apq_otc"]
                        for label in things_per_class_results
                    ]
                )
            )
            / n_things
        )
        apq = (apq_stuff * n_stuff + apq_things * n_things) / (n_stuff + n_things)

        return (
            {
                "apq": apq,
                "apq_stuff": apq_stuff,
                "apq_things": apq_things,
                "apq_vis": apq_vis,
                "apq_occ": apq_occ,
            },
            stuff_per_class_results,
            things_per_class_results,
        )


class APCStatCat:
    def __init__(self):
        self.vis_weighted_iou = 0.0
        self.vis_gt_area = 0
        self.occ_weighted_iou = 0.0
        self.occ_gt_area = 0

    def __iadd__(self, apc_stat_cat):
        self.vis_weighted_iou += apc_stat_cat.vis_weighted_iou
        self.vis_gt_area += apc_stat_cat.vis_gt_area
        self.occ_weighted_iou += apc_stat_cat.occ_weighted_iou
        self.occ_gt_area += apc_stat_cat.occ_gt_area
        return self


class APCStat:
    def __init__(self):
        self.apc_per_cat = defaultdict(APCStatCat)

    def __getitem__(self, i):
        return self.apc_per_cat[i]

    def __iadd__(self, apc_stat):
        for label, apc_stat_cat in apc_stat.apc_per_cat.items():
            self.apc_per_cat[label] += apc_stat_cat
        return self

    def apc_average(self, categories):
        stuff_per_class_results = {}
        things_per_class_results = {}
        for label_type, label_info in categories.items():
            if label_type == "stuff":
                for label_id, label in label_info.items():
                    if self.apc_per_cat[label_id].vis_gt_area > 0:
                        cov_sc = (
                            self.apc_per_cat[label_id].vis_weighted_iou
                            / self.apc_per_cat[label_id].vis_gt_area
                        )
                    else:
                        cov_sc = 0.0
                    stuff_per_class_results[label.name] = {"cov_sc": cov_sc}
            else:
                for label_id, label in label_info.items():
                    if self.apc_per_cat[label_id].vis_gt_area > 0:
                        cov_vtc = (
                            self.apc_per_cat[label_id].vis_weighted_iou
                            / self.apc_per_cat[label_id].vis_gt_area
                        )
                    else:
                        cov_vtc = 0.0
                    if self.apc_per_cat[label_id].occ_gt_area > 0:
                        cov_otc = (
                            self.apc_per_cat[label_id].occ_weighted_iou
                            / self.apc_per_cat[label_id].occ_gt_area
                        )
                    else:
                        cov_otc = 0.0
                    if (
                        self.apc_per_cat[label_id].vis_gt_area
                        + self.apc_per_cat[label_id].occ_gt_area
                        > 0
                    ):
                        cov_tc = (
                            self.apc_per_cat[label_id].vis_weighted_iou
                            + self.apc_per_cat[label_id].occ_weighted_iou
                        ) / (
                            self.apc_per_cat[label_id].vis_gt_area
                            + self.apc_per_cat[label_id].occ_gt_area
                        )
                    else:
                        cov_tc = 0.0
                    things_per_class_results[label.name] = {
                        "cov_vtc": cov_vtc,
                        "cov_otc": cov_otc,
                        "cov_tc": cov_tc,
                    }

        n_stuff, n_things = len(stuff_per_class_results), len(things_per_class_results)
        apc_stuff = (
            np.sum(
                np.array(
                    [
                        stuff_per_class_results[label]["cov_sc"]
                        for label in stuff_per_class_results
                    ]
                )
            )
            / n_stuff
        )
        apc_things = (
            np.sum(
                np.array(
                    [
                        things_per_class_results[label]["cov_tc"]
                        for label in things_per_class_results
                    ]
                )
            )
            / n_things
        )
        apc_vis = (
            np.sum(
                np.array(
                    [
                        things_per_class_results[label]["cov_vtc"]
                        for label in things_per_class_results
                    ]
                )
            )
            / n_things
        )
        apc_occ = (
            np.sum(
                np.array(
                    [
                        things_per_class_results[label]["cov_otc"]
                        for label in things_per_class_results
                    ]
                )
            )
            / n_things
        )
        apc = (apc_stuff * n_stuff + apc_things * n_things) / (n_stuff + n_things)

        return (
            {
                "apc": apc,
                "apc_stuff": apc_stuff,
                "apc_things": apc_things,
                "apc_vis": apc_vis,
                "apc_occ": apc_occ,
            },
            stuff_per_class_results,
            things_per_class_results,
        )


def compute_IoU(a, b, void_mask=None):
    tp = np.sum(np.logical_and(a != 0, b != 0))

    if void_mask is not None:
        fp = np.sum(np.logical_and(np.logical_and(a == 0, b != 0), void_mask != 255))
    else:
        fp = np.sum(np.logical_and(a == 0, b != 0))

    fn = np.sum(np.logical_and(a != 0, b == 0))
    iou = tp / (tp + fp + fn + 1e-8)
    return iou


def prepare_semantic_map(ann_i):
    sem = np.ones_like(ann_i, dtype=np.uint8) * VOID
    sem[ann_i > OFFSET] = ann_i[ann_i > OFFSET] // OFFSET
    sem[ann_i < OFFSET] = ann_i[ann_i < OFFSET]
    return sem


def compute_void_mask(gt_sem, all_categories):
    void_mask = np.zeros_like(gt_sem)
    for cls_id in np.unique(gt_sem):
        if cls_id not in all_categories:
            void_mask[gt_sem == cls_id] = 255
    return void_mask


def load_annotation(annotation_path):
    ann_i = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
    ann_json_path = annotation_path.replace("_ampano.png", "_ampano.json")
    with open(ann_json_path, "r") as file:
        ann_json = json.load(file)
    return ann_i, ann_json


class apq_compute_worker:
    def __init__(self, categories):
        self.categories = categories
        self.all_categories = {**categories["stuff"], **categories["things"]}

    def __call__(self, annotation_set):
        apq_stat = APQStat()

        gt_ann, pred_ann = annotation_set

        gt_ann_i, gt_ann_json = load_annotation(gt_ann)
        pred_ann_i, pred_ann_json = load_annotation(pred_ann)

        pred_sem = prepare_semantic_map(pred_ann_i)
        gt_sem = prepare_semantic_map(gt_ann_i)
        void_mask = compute_void_mask(gt_sem, self.all_categories)

        apq_stat = self.update_stats_for_stuff_categories(
            gt_sem, pred_sem, void_mask, apq_stat
        )
        gt_segments, pred_segments = self.prepare_segments(
            gt_ann_i, pred_ann_i, gt_ann_json, pred_ann_json
        )
        apq_stat = self.get_apq_things(gt_segments, pred_segments, void_mask, apq_stat)

        return apq_stat

    def load_annotation(self, annotation_path):
        ann_i = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
        ann_json_path = annotation_path.replace("_ampano.png", "_ampano.json")
        with open(ann_json_path, "r") as file:
            ann_json = json.load(file)
        return ann_i, ann_json

    def update_stats_for_stuff_categories(self, gt_sem, pred_sem, void_mask, apq_stat):
        for cls_id in np.unique(gt_sem):
            if cls_id not in self.categories["stuff"]:
                continue
            matched = False
            for pred_cls_id in np.unique(pred_sem):
                if pred_cls_id == cls_id:
                    iou = compute_IoU(
                        gt_sem == cls_id, pred_sem == pred_cls_id, void_mask
                    )
                    apq_stat[cls_id].iou += iou
                    apq_stat[cls_id].tp += 1
                    matched = True

            if not matched:
                apq_stat[cls_id].fn += 1
        return apq_stat

    def prepare_segments(self, gt_ann_i, pred_ann_i, gt_ann_json, pred_ann_json):
        gt_segments = {}
        for gt_idx in np.unique(gt_ann_i):
            if gt_idx < OFFSET:
                continue

            occluded = gt_ann_json[str(gt_idx)]["occluded"]
            if occluded:
                occlusion_mask_gt = mask_utils.decode(
                    gt_ann_json[str(gt_idx)]["occlusion_mask"]
                ).astype(np.uint8)
            else:
                occlusion_mask_gt = np.zeros_like(gt_ann_i).astype(np.uint8)

            visible_mask_gt = gt_ann_i == gt_idx
            if gt_ann_json[str(gt_idx)]["occluded"]:
                amodal_mask_gt = mask_utils.decode(
                    gt_ann_json[str(gt_idx)]["amodal_mask"]
                ).astype(np.uint8)
            else:
                amodal_mask_gt = visible_mask_gt.copy()

            gt_segments[gt_idx] = (
                visible_mask_gt,
                occlusion_mask_gt,
                amodal_mask_gt,
                occluded,
            )

        pred_segments = {}
        for pred_idx in np.unique(pred_ann_i):
            if pred_idx < OFFSET:
                continue

            visible_mask_pred = pred_ann_i == pred_idx
            amodal_mask_pred = mask_utils.decode(
                pred_ann_json[str(pred_idx)]["amodal_mask"]
            ).astype(np.uint8)
            if len(pred_ann_json[str(pred_idx)]["occlusion_mask"]) > 0:
                occlusion_mask_pred = mask_utils.decode(
                    pred_ann_json[str(pred_idx)]["occlusion_mask"]
                ).astype(np.uint8)
            else:
                occlusion_mask_pred = np.logical_and(
                    amodal_mask_pred, np.logical_not(visible_mask_pred)
                ).astype(np.uint8)
            occluded = np.sum(occlusion_mask_pred) > 0
            pred_segments[pred_idx] = (
                visible_mask_pred,
                occlusion_mask_pred,
                amodal_mask_pred,
                occluded,
            )

        return gt_segments, pred_segments

    def get_apq_things(self, gt_segments, pred_segments, void_mask, apq_stat):
        if len(gt_segments) == 0 and len(pred_segments) == 0:
            return apq_stat

        elif len(gt_segments) > 0 and len(pred_segments) == 0:
            for i, gt_idx in enumerate(gt_segments):
                gt_cls = gt_idx // OFFSET
                occ_flag = gt_segments[gt_idx][-1]
                apq_stat[gt_cls].fn += 1
                if occ_flag:
                    apq_stat[gt_cls].occ_fn += 1

                return apq_stat

        elif len(gt_segments) == 0 and len(pred_segments) > 0:
            for j, pred_idx in enumerate(pred_segments):
                vis_mask = pred_segments[pred_idx][0]
                vis_void_iou = compute_IoU(void_mask, vis_mask)
                if vis_void_iou > 0.5:
                    continue
                pred_cls = pred_idx // OFFSET
                apq_stat[pred_cls].fp += 1
                if pred_segments[pred_idx][-1]:
                    apq_stat[pred_cls].occ_fp += 1

                return apq_stat

        cost_matrix = np.ones((len(gt_segments), len(pred_segments), 3)) * 100
        gt_idx_map = {}
        pred_idx_map = {}
        for i, gt_idx in enumerate(gt_segments):
            gt_visible, gt_occlusion, gt_amodal, occ_flag = gt_segments[gt_idx]
            gt_cls = gt_idx // OFFSET
            gt_idx_map[i] = gt_idx
            for j, pred_idx in enumerate(pred_segments):
                pred_visible, pred_occlusion, pred_amodal, _ = pred_segments[pred_idx]
                pred_cls = pred_idx // OFFSET
                iou_amodal = compute_IoU(gt_amodal, pred_amodal)
                iou_vis = compute_IoU(gt_visible, pred_visible)
                iou_occ = compute_IoU(gt_occlusion, pred_occlusion)
                pred_idx_map[j] = pred_idx
                if gt_cls == pred_cls:
                    cost_matrix[i, j, 0] = 1 - iou_amodal
                    cost_matrix[i, j, 1] = iou_vis
                    cost_matrix[i, j, 2] = iou_occ

        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix[:, :, 0])
        fp_col_ind = np.setdiff1d(np.arange(cost_matrix.shape[1]), col_ind)
        for row_ind_i, col_ind_i in zip(row_ind, col_ind):
            matching_cost = cost_matrix[row_ind_i, col_ind_i, 0]
            gt_cls = gt_idx_map[row_ind_i] // OFFSET
            occ_flag = gt_segments[gt_idx_map[row_ind_i]][-1]
            pred_occ_flag = pred_segments[pred_idx_map[col_ind_i]][-1]
            if matching_cost == 100:
                apq_stat[gt_cls].fn += 1
                if occ_flag:
                    apq_stat[gt_cls].occ_fn += 1
            else:
                apq_stat[gt_cls].tp += 1
                apq_stat[gt_cls].iou += cost_matrix[row_ind_i, col_ind_i, 1]
                if occ_flag:
                    apq_stat[gt_cls].occ_tp += 1
                    apq_stat[gt_cls].occ_iou += cost_matrix[row_ind_i, col_ind_i, 2]
                elif pred_occ_flag:
                    apq_stat[gt_cls].occ_fp += 1

        for col_ind_i in fp_col_ind:
            pred_cls = pred_idx_map[col_ind_i] // OFFSET
            apq_stat[pred_cls].fp += 1
            if pred_segments[pred_idx][-1]:
                apq_stat[pred_cls].occ_fp += 1

        return apq_stat


class apc_compute_worker:
    def __init__(self, categories):
        self.categories = categories
        self.all_categories = {**categories["stuff"], **categories["things"]}

    def __call__(self, annotation_set):
        apc_stat = APCStat()

        gt_ann, pred_ann = annotation_set
        gt_ann_i, gt_ann_json = load_annotation(gt_ann)
        pred_ann_i, pred_ann_json = load_annotation(pred_ann)

        pred_sem = prepare_semantic_map(pred_ann_i)
        gt_sem = prepare_semantic_map(gt_ann_i)
        void_mask = compute_void_mask(gt_sem, self.all_categories)
        apc_stat = self.update_stats_for_stuff_categories(
            gt_sem, pred_sem, void_mask, apc_stat
        )
        apc_stat = self.update_stats_for_thing_categories(
            gt_ann_i, pred_ann_i, gt_ann_json, pred_ann_json, void_mask, apc_stat
        )

        return apc_stat

    def update_stats_for_stuff_categories(self, gt_sem, pred_sem, void_mask, apc_stat):
        for cls_id in np.unique(gt_sem):
            ious = [
                0.0,
            ]
            if cls_id not in self.categories["stuff"]:
                continue
            for pred_cls_id in np.unique(pred_sem):
                if pred_cls_id == cls_id:
                    iou = compute_IoU(
                        gt_sem == cls_id, pred_sem == pred_cls_id, void_mask
                    )
                    ious.append(iou)

            max_ind = np.argmax(np.array(ious))
            gt_area = np.sum(gt_sem == cls_id)
            weighted_iou = ious[max_ind] * gt_area

            apc_stat[cls_id].vis_weighted_iou += weighted_iou
            apc_stat[cls_id].vis_gt_area += gt_area

        return apc_stat

    def update_stats_for_thing_categories(
        self, gt_ann_i, pred_ann_i, gt_ann_json, pred_ann_json, void_mask, apc_stat
    ):
        pred_segments = {}
        for pred_idx in np.unique(pred_ann_i):
            if pred_idx < OFFSET:
                continue

            visible_mask_pred = pred_ann_i == pred_idx
            if len(pred_ann_json[str(pred_idx)]["occlusion_mask"]) > 0:
                occlusion_mask_pred = mask_utils.decode(
                    pred_ann_json[str(pred_idx)]["occlusion_mask"]
                ).astype(np.uint8)
            else:
                amodal_mask_pred = mask_utils.decode(
                    pred_ann_json[str(pred_idx)]["amodal_mask"]
                ).astype(np.uint8)
                occlusion_mask_pred = np.logical_and(
                    amodal_mask_pred, np.logical_not(visible_mask_pred)
                ).astype(np.uint8)
            pred_segments[pred_idx] = (visible_mask_pred, occlusion_mask_pred)

        for gt_idx in np.unique(gt_ann_i):
            if gt_idx < OFFSET:
                continue

            cls_gt = gt_idx // OFFSET
            ious_vis = [
                0.0,
            ]
            ious_occ = [
                0.0,
            ]

            if gt_ann_json[str(gt_idx)]["occluded"]:
                occlusion_mask_gt = mask_utils.decode(
                    gt_ann_json[str(gt_idx)]["occlusion_mask"]
                ).astype(np.uint8)
            else:
                occlusion_mask_gt = np.zeros_like(gt_ann_i).astype(np.uint8)

            visible_mask_gt = gt_ann_i == gt_idx

            for pred_idx in pred_segments:

                cls_pred = pred_idx // OFFSET
                iou_vis = compute_IoU(
                    visible_mask_gt, pred_segments[pred_idx][0], void_mask
                )
                iou_occ = compute_IoU(occlusion_mask_gt, pred_segments[pred_idx][1])
                if cls_gt == cls_pred:
                    ious_vis.append(iou_vis)
                    ious_occ.append(iou_occ)

            max_ind = np.argmax(np.array(ious_vis))
            gt_vis_area = np.sum(visible_mask_gt)
            gt_occ_area = np.sum(occlusion_mask_gt)
            weighted_iou_vis = ious_vis[max_ind] * gt_vis_area
            weighted_iou_occ = ious_occ[max_ind] * gt_occ_area
            apc_stat[cls_gt].vis_weighted_iou += weighted_iou_vis
            apc_stat[cls_gt].vis_gt_area += gt_vis_area
            apc_stat[cls_gt].occ_weighted_iou += weighted_iou_occ
            apc_stat[cls_gt].occ_gt_area += gt_occ_area

        return apc_stat


def apc_compute_multi_core(matched_annotations_list, categories, cpu_num=-1):
    if cpu_num == -1:
        cpu_num = multiprocessing.cpu_count()

    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print(
        "Evaluating APC with Number of cores: {}, images per core: {}".format(
            cpu_num, len(annotations_split[0])
        )
    )
    apc_workers = apc_compute_worker(categories)
    apc_stat = APCStat()

    with multiprocessing.Pool(processes=cpu_num) as pool:
        total = len(matched_annotations_list)
        for apc_stat_i in tqdm.tqdm(
            pool.imap(apc_workers, matched_annotations_list, cpu_num), total=total
        ):
            apc_stat += apc_stat_i

    return apc_stat


def apq_compute_multi_core(matched_annotations_list, categories, cpu_num=-1):
    if cpu_num == -1:
        cpu_num = multiprocessing.cpu_count()

    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print(
        "Evaluating APQ with Number of cores: {}, images per core: {}".format(
            cpu_num, len(annotations_split[0])
        )
    )
    apq_workers = apq_compute_worker(categories)
    apq_stat = APQStat()

    with multiprocessing.Pool(processes=cpu_num) as pool:
        total = len(matched_annotations_list)
        for apq_stat_i in tqdm.tqdm(
            pool.imap(apq_workers, matched_annotations_list, cpu_num), total=total
        ):
            apq_stat += apq_stat_i

    return apq_stat


def print_results_apc(results, stuff_classes, things_classes):
    print("Stuff classes:")
    print("{:14s}| {:s}".format("Category", "Cov_sc"))
    for name, value in stuff_classes.items():
        cov_sc = "{:.2f}".format(round(value["cov_sc"] * 100, 2))
        print("{:14s}| {:7s}".format(name, cov_sc.zfill(5)))

    print("")
    print("Things classes:")
    print("{:14s}| {:s} {:s} {:s}".format("Category", "Cov_vtc", "Cov_otc", "Cov_tc"))
    for name, value in things_classes.items():
        cov_vtc = "{:.2f}".format(round(value["cov_vtc"] * 100, 2))
        cov_otc = "{:.2f}".format(round(value["cov_otc"] * 100, 2))
        cov_tc = "{:.2f}".format(round(value["cov_tc"] * 100, 2))
        print(
            "{:14s}| {:7s} {:7s} {:7s}".format(
                name, str(cov_vtc).zfill(5), str(cov_otc).zfill(5), str(cov_tc).zfill(5)
            )
        )

    print("")
    print("APC results:")
    for name, value in results.items():
        print("{:14s}| {:.2f}".format(name.capitalize(), round(value * 100, 2)))


def print_results_apq(results, stuff_classes, things_classes):
    print("Stuff classes:")
    print("{:14s}| {:s}".format("Category", "APQ_sc"))
    for name, value in stuff_classes.items():
        apq_sc = "{:.2f}".format(round(value["apq_sc"] * 100, 2))
        print("{:14s}| {:7s}".format(name, apq_sc.zfill(5)))

    print("")
    print("Things classes:")
    print("{:14s}| {:s} {:s} {:s}".format("Category", "APQ_vtc", "APQ_otc", "APQ_tc"))
    for name, value in things_classes.items():
        apq_vtc = "{:.2f}".format(round(value["apq_vtc"] * 100, 2))
        apq_otc = "{:.2f}".format(round(value["apq_otc"] * 100, 2))
        apq_tc = "{:.2f}".format(round(value["apq_tc"] * 100, 2))
        print(
            "{:14s}| {:7s} {:7s} {:7s}".format(
                name, str(apq_vtc).zfill(5), str(apq_otc).zfill(5), str(apq_tc).zfill(5)
            )
        )

    print("")
    print("APQ results:")
    for name, value in results.items():
        print("{:14s}| {:.2f}".format(name.capitalize(), round(value * 100, 2)))


def find_files(folder):
    file_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith("_ampano.png"):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    return file_paths


def get_categories(datasetName):
    if datasetName == "amodalSynthDrive":
        id2label = asd_id2label
    else:
        raise ValueError("Unknown dataset name: {}".format(datasetName))

    categories = {}
    categories["stuff"] = {
        id2label[label].id: id2label[label]
        for label in id2label
        if not id2label[label].hasInstances and not id2label[label].ignoreInEval
    }
    categories["things"] = {
        id2label[label].id: id2label[label]
        for label in id2label
        if id2label[label].hasInstances and not id2label[label].ignoreInEval
    }
    return categories


def evaluate(
    dataset_name, gt_folder, pred_folder, results_file, cpu_num=-1
):

    start_time = time.time()
    gt_files = find_files(gt_folder)
    pred_files = [el.replace(gt_folder, pred_folder) for el in gt_files]

    print("Evaluation amodal panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))

    if not os.path.isdir(gt_folder):
        print_error(
            "Folder {} with ground truth segmentations doesn't exist".format(gt_folder)
        )
    if not os.path.isdir(pred_folder):
        print_error(
            "Folder {} with predicted segmentations doesn't exist".format(pred_folder)
        )

    matched_annotations_list = []
    for gt_ann, pred_ann in zip(gt_files, pred_files):
        if not os.path.exists(pred_ann):
            raise Exception("no prediction for the groundtruth: {}".format(gt_ann))
        if not os.path.exists(pred_ann.replace("_ampano.png", "_ampano.json")):
            raise Exception(
                "no json prediction file for the groundtruth: {}".format(gt_ann)
            )
        matched_annotations_list.append((gt_ann, pred_ann))

    categories = get_categories(dataset_name)
    apq_stat = apq_compute_multi_core(matched_annotations_list, categories, cpu_num)
    apq_results, apq_stuff_classes, apq_things_classes = apq_stat.apq_average(
        categories
    )
    print_results_apq(apq_results, apq_stuff_classes, apq_things_classes)

    apc_stat = apc_compute_multi_core(matched_annotations_list, categories, cpu_num)
    apc_results, apc_stuff_classes, apc_things_classes = apc_stat.apc_average(
        categories
    )

    print_results_apc(apc_results, apc_stuff_classes, apc_things_classes)

    with open(results_file, "w") as f:
        print("Saving computed results in {}".format(results_file))
        for results in [apc_results, apc_stuff_classes, apc_things_classes]:
            json.dump(results, f, sort_keys=True, indent=4)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return apq_results, apc_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-folder",
        dest="gt_folder",
        help="""path of amodal_panoptic_seg folder that contains
                            ground truth *_ampano.png and *_ampano.json files. 
                        """,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--prediction-folder",
        dest="prediction_folder",
        help="""path of amodal_panoptic_seg folder that contains
                            prediction *_ampano.png and *_ampano.json files. 
                        """,
        default=None,
        type=str,
    )
    resultFile = "result_amodal_panoptic.json"
    parser.add_argument(
        "--results_file",
        dest="results_file",
        help="File to store computed panoptic quality. Default: {}".format(resultFile),
        default=resultFile,
        type=str,
    )
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="""name of the dataset. Default: amodalSynthDrive.
                    """,
        default="amodalSynthDrive",
        type=str,
    )
    parser.add_argument(
        "--cpu-num",
        dest="cpu_num",
        help="""number of CPU cores to use for evaluation. Default: -1 (use all available cores).
                    """,
        default=-1,
        type=int,
    )
    args = parser.parse_args()

    apq_results, apc_results = evaluate(
        args.dataset_name,
        args.gt_folder,
        args.prediction_folder,
        args.results_file,
        args.cpu_num,
    )

    return apq_results, apc_results


if __name__ == "__main__":
    main()
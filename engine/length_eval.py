import time

import torch
from transformers import AutoTokenizer


class AverageMeter(object):
    """Store and compute average/value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def compute_batch_iou_detailed(pred, gt):
    """
    Args:
        pred: [B, 2, H, W]
        gt:   [B, H, W]
    Returns:
        intersection: [B]
        union:        [B]
        ious:         [B]
    """
    pred_class = pred.argmax(dim=1)
    pred_flat = pred_class.view(pred_class.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)

    intersection = torch.logical_and(pred_flat == 1, gt_flat == 1).sum(dim=1).float()
    union = torch.logical_or(pred_flat == 1, gt_flat == 1).sum(dim=1).float()
    ious = intersection / (union + 1e-6)
    return intersection, union, ious


def decode_segment_text(segment_word_ids, tokenizer):
    """
    Decode one segment IDs into plain text.
    """
    if torch.is_tensor(segment_word_ids):
        token_ids = segment_word_ids.detach().cpu().tolist()
    else:
        token_ids = list(segment_word_ids)

    # Remove padding and special tokens ([PAD]=0, [CLS]=101, [SEP]=102 for bert-base-uncased)
    valid_token_ids = [tid for tid in token_ids if tid not in [0, 101, 102]]
    if not valid_token_ids:
        return ""

    tokens = tokenizer.convert_ids_to_tokens(valid_token_ids)
    return tokenizer.convert_tokens_to_string(tokens).strip()


def get_query_max_segment_word_length(word_ids_2d, tokenizer):
    """
    word_ids_2d shape: [num_segments, max_tokens]
    Length definition: max word-count among all decoded segments.
    """
    if torch.is_tensor(word_ids_2d):
        segment_count = word_ids_2d.size(0)
    else:
        segment_count = len(word_ids_2d)

    max_word_len = 0
    segment_texts = []
    for seg_idx in range(segment_count):
        seg_text = decode_segment_text(word_ids_2d[seg_idx], tokenizer)
        segment_texts.append(seg_text)
        if seg_text:
            seg_word_len = len([w for w in seg_text.split() if w])
            if seg_word_len > max_word_len:
                max_word_len = seg_word_len

    merged_text = " ".join([s for s in segment_texts if s]).strip()
    return max_word_len, merged_text, segment_texts


def get_length_bucket(length_value, length_bins):
    for lo, hi in length_bins:
        if lo <= length_value <= hi:
            return f"[{lo},{hi}]"
    lo, hi = length_bins[-1]
    return f"[{lo},{hi}]"


def validate_epoch_by_length_bins(args, val_loader, model, logger, mode="val", val_epoch=99, length_bins=None):
    """
    Extra evaluation for RISBench val:
    - group samples by query length
    - query length = max word-count among text segments
    """
    logger.info(f"--- Starting {mode} Epoch {val_epoch} (Length-Bin Eval) ---")
    device = torch.cuda.current_device()
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    if length_bins is None:
        length_bins = [(0, 9), (9, 12), (12, 15), (15, 20)]

    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    overall_miou_meter = AverageMeter()
    overall_inter = 0.0
    overall_union = 0.0
    overall_prec = {t: AverageMeter() for t in prec_thresholds}

    bucket_stats = {}
    for lo, hi in length_bins:
        key = f"[{lo},{hi}]"
        bucket_stats[key] = {
            "count": 0,
            "inter": 0.0,
            "union": 0.0,
            "miou": AverageMeter(),
            "prec": {t: AverageMeter() for t in prec_thresholds},
        }

    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    torch.cuda.empty_cache()

    for batch_idx, (imgs, seg_map, word_id, mask) in enumerate(val_loader):
        imgs = imgs.to(device)
        seg_map = seg_map.to(device)
        word_id = word_id.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            num_descs = word_id.size(-1)
            for j in range(num_descs):
                mask_out, _ = model(imgs, word_id[:, :, :, j], mask[:, :, :, j])
                intersections, unions, ious = compute_batch_iou_detailed(mask_out, seg_map)

                batch_size = imgs.size(0)
                overall_inter += intersections.sum().item()
                overall_union += unions.sum().item()
                overall_miou_meter.update(ious.mean().item(), batch_size)
                for t in prec_thresholds:
                    overall_prec[t].update((ious > t).float().mean().item(), batch_size)

                for k in range(batch_size):
                    # Use the longest decoded segment as query length.
                    q_len, _, _ = get_query_max_segment_word_length(word_id[k, :, :, j], tokenizer)
                    bucket_key = get_length_bucket(q_len, length_bins)
                    bucket = bucket_stats[bucket_key]

                    sample_iou = ious[k].item()
                    bucket["count"] += 1
                    bucket["inter"] += intersections[k].item()
                    bucket["union"] += unions[k].item()
                    bucket["miou"].update(sample_iou, 1)
                    for t in prec_thresholds:
                        bucket["prec"][t].update(1.0 if sample_iou > t else 0.0, 1)

        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 1000 == 0:
            logger.info(
                f"[LengthEval {batch_idx}/{len(val_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"mIoU {overall_miou_meter.val:.4f} ({overall_miou_meter.avg:.4f})"
            )

    final_miou = overall_miou_meter.avg
    final_oiou = overall_inter / (overall_union + 1e-6)
    final_prec = {t: overall_prec[t].avg for t in prec_thresholds}

    logger.info("=" * 40)
    logger.info(f"FINAL {mode.upper()} RESULTS (ALL):")
    logger.info(f"mIoU: {final_miou:.4f}")
    logger.info(f"oIoU: {final_oiou:.4f}")
    for t in prec_thresholds:
        logger.info(f"Pr@{int(t * 100)}: {final_prec[t]:.4f}")
    logger.info("-" * 40)
    logger.info("LENGTH BUCKET RESULTS:")

    bucket_report = {}
    for lo, hi in length_bins:
        key = f"[{lo},{hi}]"
        stats = bucket_stats[key]
        count = stats["count"]

        if count == 0:
            logger.info(f"{key} words | count=0 | no samples")
            bucket_report[key] = {
                "count": 0,
                "mIoU": 0.0,
                "oIoU": 0.0,
                **{f"Pr@{int(t * 100)}": 0.0 for t in prec_thresholds},
            }
            continue

        b_miou = stats["miou"].avg
        b_oiou = stats["inter"] / (stats["union"] + 1e-6)
        b_prec = {t: stats["prec"][t].avg for t in prec_thresholds}
        logger.info(f"{key} words | count={count} | mIoU={b_miou:.4f} | oIoU={b_oiou:.4f}")
        for t in prec_thresholds:
            logger.info(f"  Pr@{int(t * 100)}: {b_prec[t]:.4f}")

        bucket_report[key] = {
            "count": count,
            "mIoU": b_miou,
            "oIoU": b_oiou,
            **{f"Pr@{int(t * 100)}": b_prec[t] for t in prec_thresholds},
        }

    logger.info("=" * 40)
    return final_miou, final_oiou, final_prec, bucket_report

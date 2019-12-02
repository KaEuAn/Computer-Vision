
def check(a):
    return max(a, 0)


def square(bbox):
    return check(bbox[2] - bbox[0]) * check(bbox[3] - bbox[1])


def iou_score(first_bbox, second_bbox):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(first_bbox) == 4
    assert len(second_bbox) == 4

    bbox = [max(first_bbox[0], second_bbox[0]), max(first_bbox[1], second_bbox[1]), min(first_bbox[2], second_bbox[2]),
            min(first_bbox[3], second_bbox[3])]
    intersection = square(bbox)
    return intersection / (square(first_bbox) + square(second_bbox) - intersection)

    return ...

def convert_to_dict(detections):
    last_detected = {}
    for det in detections:
        last_detected[det[0]] = det[1:]
    return last_detected

def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = convert_to_dict(frame_obj)
        frame_hyp = convert_to_dict(frame_hyp)
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_matches = {}
        for key, key_i in matches.items():
            if key in frame_obj.keys() and key_i in frame_hyp.keys():
                score = iou_score(frame_obj[key], frame_hyp[key_i])
                if score > threshold:
                    dist_sum += score
                    match_count += 1
                    del frame_hyp[key_i]
                    del frame_obj[key]
                    new_matches[key] = key_i
        matches = new_matches

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: Update matches with current matched IDs
        pairs = []  # (score, track_id, index in detections)
        for key, prev_detect in frame_obj.items():
            for key_i, now_detect in frame_hyp.items():
                score = iou_score(now_detect, prev_detect)
                pairs.append([score, key, key_i])
        pairs = sorted(pairs, reverse=True)
        for pair in pairs:
            if pair[1] in frame_obj.keys() and pair[2] in frame_hyp.keys():
                score = iou_score(frame_obj[pair[1]], frame_hyp[pair[2]])
                if score > threshold:
                    del frame_obj[pair[1]]
                    del frame_hyp[pair[2]]
                    matches[pair[1]] = pair[2]

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        pass

    # Step 8: Calculate MOTP and MOTA
    MOTP = ...
    MOTA = ...

    return MOTP, MOTA

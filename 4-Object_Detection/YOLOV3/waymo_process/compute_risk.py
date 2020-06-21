import math

from waymo_open_dataset import dataset_pb2 as open_dataset

'''
NOTE:
    1) Provided by Ayoosh Bansal
    2) Source: https://gitlab.engr.illinois.edu/rtesl/synergistic_redundancy/evaluation/detect_evaluate/-/blob/master/waymo_utils.py#L83
'''

waymo_to_coco = {0: 10, 1: 1, 2: 0, 3: 8, 4: 5}  # from waymo to coco


class Bb2d:
    '''
    Bounding Box 2D class
    '''

    def __init__(self, x1, y1, x2, y2, cat, conf=1, rr=1, id=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cat = cat
        self.conf = conf
        self.rr = rr
        self.id = id

    def __str__(self):
        return str(self.x1) + ' ' + str(self.y1) + ' ' + str(self.x2) + ' ' + str(self.y2) + ', cat:' + str(self.cat)


def waymo_label_to_BB(label, risk=-1):
    return Bb2d(float(label.box.center_x - 0.5 * label.box.length),
                float(label.box.center_y - 0.5 * label.box.width),
                float(label.box.center_x + 0.5 * label.box.length),
                float(label.box.center_y + 0.5 * label.box.width),
                int(waymo_to_coco[label.type]), float(1), risk, label.id)


def bb_intersection_over_union(boxA, boxB):
    '''
    Intersection over union of 2 Bb2d
    '''
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA.x2 - boxA.x1 + 1) * (boxA.y2 - boxA.y1 + 1)
    boxBArea = (boxB.x2 - boxB.x1 + 1) * (boxB.y2 - boxB.y1 + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def bb_intersection_over_groundtruth(boxA, boxB):
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA.x2 - boxA.x1 + 1) * (boxA.y2 - boxA.y1 + 1)
    boxBArea = (boxB.x2 - boxB.x1 + 1) * (boxB.y2 - boxB.y1 + 1)
    iog = interArea / float(boxAArea)
    return iog


def find_best_match(target_bbs, bb, rrr=False):
    '''
    Given the groundtruth camera bbx and one projected lidar bbx, it returns the maximum iou obtained
    and the index of the gt bbx.
    '''
    iou = []
    for tbb in target_bbs:
        if rrr:
            iou.append(bb_intersection_over_groundtruth(tbb, bb))
        else:
            iou.append(bb_intersection_over_union(tbb, bb))

    if not iou:
        return 0, -1

    iou_max = max(iou)
    i = iou.index(iou_max)
    return iou_max, i


def waymo_find_best_match_id(frame, camera_label, rrr=False, iou_thresh=0.5, use_single_camera=False):
    '''
    Find the best match laser label ID for given camera label.
    Use the projected lidar labels to find overlapps with camera labels.
    '''
    # process front camera only
    if use_single_camera:
        target_camera_names = {open_dataset.CameraName.FRONT}
    else:
        target_camera_names = {open_dataset.CameraName.FRONT,
                               open_dataset.CameraName.FRONT_LEFT,
                               open_dataset.CameraName.FRONT_RIGHT,
                               open_dataset.CameraName.SIDE_LEFT,
                               open_dataset.CameraName.SIDE_RIGHT}

    target_bbs = []
    for projected_labels in frame.projected_lidar_labels:
        # skip unwanted cameras
        if projected_labels.name not in target_camera_names:
            continue

        for label in projected_labels.labels:
            target_bbs.append(waymo_label_to_BB(label))

    bb = waymo_label_to_BB(camera_label)
    iou_max, i = find_best_match(target_bbs, bb, rrr)

    if iou_max >= iou_thresh:
        # return target_bbs[i].id.split('_')[0]
        return target_bbs[i].id[0:22]
    else:
        return None


def compute_risk_for_laser_objects(frame, acc_to_brake=-2.5, t_reaction=5):
    '''
    Compute the risk for all objects within a frame.
    '''
    # laser object id --> risk
    laser_object_risks = dict()

    # compute velocity
    vel = 0
    for img in frame.images:
        vel += math.sqrt(pow(img.velocity.v_x, 2) + pow(img.velocity.v_y, 2) + pow(img.velocity.v_z, 2))
    vel /= len(frame.images)

    # compute braking distance
    s_braking = -pow(vel, 2) / (2 * acc_to_brake)

    # compute reaction distance
    s_reaction = vel * t_reaction

    # compute stop distance
    s = s_braking + s_reaction

    for label in frame.laser_labels:
        # label_id = label.id.split('_')[0]
        label_id = label.id[0:22]

        # compute distance
        distance = math.sqrt(pow(label.box.center_x, 2) + pow(label.box.center_y, 2))
        distance -= 4  # workaround to guess nearest surface of the object. XXX

        # assign R
        if distance <= s:
            R = 1
        else:
            R = s / distance

        laser_object_risks[label_id] = R

    return laser_object_risks

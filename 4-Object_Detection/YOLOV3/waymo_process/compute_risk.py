import math

'''
NOTE:
    1) Provided by Ayoosh Bansal
    2) Source: https://gitlab.engr.illinois.edu/rtesl/synergistic_redundancy/evaluation/detect_evaluate/-/blob/master/waymo_utils.py#L83
'''


def compute_risk_for_objects(frame, acc_to_brake=-2.5, t_reaction=5):
    '''
    Compute the risk for all objects within a frame.
    '''
    object_risks = dict()

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
        label_id = label.id
        # print(label_id)

        # compute distance
        distance = math.sqrt(pow(label.box.center_x, 2) + pow(label.box.center_y, 2))
        distance -= 4  # workaround to guess nearest surface of the object. XXX

        # assign R
        if distance <= s:
            R = 1
        else:
            R = s / distance

        object_risks[label_id] = R

    return object_risks

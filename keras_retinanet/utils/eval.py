"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .anchors import compute_overlap
from .visualization import draw_bbox_detections, draw_bbox_annotations, draw_pose_detections

import keras
import numpy as np
import os
import time
import csv

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# RotinaNet-6D
def _test_pose_ADD(gt_pose_translation, gt_pose_rotation, detected_pose_translation,
            detected_pose_rotation, point_cloud_test, distance_diag, diag_threshold, print_depth=False):
    """ Compute the mean average distance between the gt and predeicted points clouds. 

    # Arguments
        gt_pose_translation,        :
        gt_pose_rotation,           :
        detected_pose_translation,  :
        detected_pose_rotation,     :
        point_cloud_test,           :
        distance_diag,              :
        diag_threshold,             :
        print_depth=False           :
    # Returns
        Mean Averge Distance, accepted (bool)
    """
    gt_pose_rotation = gt_pose_rotation.reshape((3,3))
    detected_pose_rotation = np.transpose(detected_pose_rotation.reshape((3,3))) # Not known why the transpose is necesarry - FIGURE OUT!!
    #detected_pose_rotation = detected_pose_rotation.reshape((3,3))

    U, _, V_t = np.linalg.svd(detected_pose_rotation, full_matrices=True)
    det = np.sign(np.linalg.det(np.matmul(V_t.T,U.T)))
    detected_pose_rotation = np.matmul(np.matmul(V_t.T, np.array([[1,0,0],[0,1,0],[0,0,det]])), U.T)

    newPL_ori = np.transpose( np.matmul(gt_pose_rotation, np.transpose(point_cloud_test)) )
    newPL_ori = newPL_ori + gt_pose_translation #+ np.tile(np.array(gt_pose_translation), (38, 1))

    newPL = np.transpose( np.matmul(detected_pose_rotation, np.transpose(point_cloud_test)) )
    newPL = newPL + detected_pose_translation #+ np.tile(np.array(detected_pose_translation), (38, 1))


    calc = np.sqrt( np.sum( (newPL - newPL_ori) * (newPL - newPL_ori), axis = 1) )
    meanAvgDist = np.mean( calc )

    if print_depth == True:
        pred_file = open("depth_preds.csv", "a")
        pred_writer = csv.writer(pred_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        pred_writer.writerow([gt_pose_translation[2], detected_pose_translation[2]])
        pred_file.close()
    
    if( meanAvgDist < distance_diag*diag_threshold):
        return meanAvgDist, 1
    else:
        return meanAvgDist, 0
    return


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_bbox_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_rotation_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_translation_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    all_inferences = [None for i in range(generator.size())]

    # Create subdirectories in save path for bbox and pose visualizations
    if save_path is not None and not os.path.exists(save_path + '/bbox'):
        os.makedirs(save_path + '/bbox')
    if save_path is not None and not os.path.exists(save_path + '/pose'):
        os.makedirs(save_path + '/pose')

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        start = time.time()
        boxes, scores, labels, rotations, translations = model.predict_on_batch(np.expand_dims(image, axis=0))[:5] #RotinaNet-6D #[:3]
        inference_time = time.time() - start

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes         = boxes[0, indices[scores_sort], :]
        image_rotations     = rotations[0,indices[scores_sort],:]
        image_translations  = translations[0,indices[scores_sort],:]
        image_scores        = scores[scores_sort]
        image_labels        = labels[0, indices[scores_sort]]
        image_detections    = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            # Copy() is necessary, else the boxes will not be printed on the saved images.
            draw_image_bbox = raw_image.copy()
            draw_bbox_annotations(draw_image_bbox, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_bbox_detections(draw_image_bbox, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)
            cv2.imwrite(os.path.join(save_path + '/bbox', '{}.png'.format(i)), draw_image_bbox)
            
            # RotinaNet-6D -> Print predicted bbox + depth prediction. 
            draw_image_pose = raw_image.copy()
            draw_pose_detections(draw_image_pose, image_boxes, image_scores, image_labels, image_rotations, image_translations, label_to_name=generator.label_to_name)
            cv2.imwrite(os.path.join(save_path + '/pose', '{}.png'.format(i)), draw_image_pose)


        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_bbox_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            all_translation_detections[i][label] = image_translations[image_detections[:, -1] == label, :]
            all_rotation_detections[i][label] = image_rotations[image_detections[:, -1] == label, :]

        all_inferences[i] = inference_time

    return all_bbox_detections, all_translation_detections, all_rotation_detections, all_inferences


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_bbox = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_rotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_translations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_bbox[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()
            all_rotations[i][label] = annotations['rotations'][annotations['labels'] == label, :].copy()
            all_translations[i][label] = annotations['translations'][annotations['labels'] == label, :].copy()

    return all_bbox, all_rotations, all_translations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    diag_threshold=0.1,
    score_threshold=0.05,
    max_detections=100,
    save_path=None, 
    print_depth_data=False

):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_bbox_detections, all_translation_detections, all_rotation_detections, all_inferences = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_bbox_annotations, all_rotation_annotations, all_translation_annotations  = _get_annotations(generator)
    average_precisions = {}
    CEP_ratios = {} # CEP -> Correctly Estimated Poses 
    mean_avg_distances = {}
    #print('all bbox', all_bbox_detections, 'all_translations', all_translation_detections, 'all_rotation_detections', all_rotation_detections)
    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))


    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        accepted_ADD_annotations = 0
        total_detections = 0

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        avg_distances = []

        for i in range(generator.size()):
            bbox_detections         = all_bbox_detections[i][label]
            rotation_detections     = all_rotation_detections[i][label]
            translation_detections  = all_translation_detections[i][label]
            bbox_annotations        = all_bbox_annotations[i][label]
            rotation_annotations    = all_rotation_annotations[i][label]
            translation_annotations = all_translation_annotations[i][label]

            num_annotations         += bbox_annotations.shape[0]
            detected_annotations    = []

            for idx, (d, r, t) in enumerate(zip(bbox_detections, rotation_detections, translation_detections)):
                scores = np.append(scores, d[4])

                if bbox_annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), bbox_annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

                # Evaluate pose predictions 
                    # FUTURE: Change to accomodate multiple objects of same class i one image.
                if idx == 0:  # Only evaluate top-1 prediction # RotinaNet-6D 
                    pt_cloud, diag_distance = generator.name_to_pt_cloud(generator.label_to_name(label))

                    # translation vector coordinates
                    t_z = (t[0] * 0.4219 + 0.6549) * 1000 # convert from m to mm
                    trans = np.array([translation_annotations[0][0]*1000, translation_annotations[0][1]*1000, t_z])
                    
                    anno_trans = np.array([translation_annotations[0][0], translation_annotations[0][1], translation_annotations[0][2]*0.4219 + 0.6549 ])*1000
                    #print("trans", trans)
                    #avg_dist, accepted_dist = _test_ADD(translation_annotations[0] * 1000, rotation_annotations[0], trans, r, pt_cloud, diag_distance, diag_threshold)
                    avg_dist, accepted_dist = _test_pose_ADD(anno_trans, rotation_annotations[0], trans, r, pt_cloud, diag_distance, diag_threshold, print_depth=print_depth_data)
                    avg_distances.append(avg_dist)

                    if accepted_dist:
                        accepted_ADD_annotations += 1
                    total_detections += 1
        CEP_ratio = accepted_ADD_annotations / np.maximum(total_detections, np.finfo(np.float64).eps)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        CEP_ratios[label] = CEP_ratio
        mean_avg_distances[label] = np.mean(avg_distances)

    # inference time
    inference_time = np.sum(all_inferences) / generator.size()

    return average_precisions, CEP_ratios, mean_avg_distances, inference_time

"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
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

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
from collections import OrderedDict


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = OrderedDict()
    pt_cloud_paths = OrderedDict()   # RotinaNet-6D
    distances = OrderedDict()    # RotinaNet-6D
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id, pt_cloud_path, distance = row  
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id,path_to_point_cloud,diag_distance\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))
        distance = _parse(distance, float, 'line {}: malformed distance: {{}}'.format(line)) # RotinaNet-6D

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
        pt_cloud_paths[class_name] = pt_cloud_path  # RotinaNet-6D
        distances[class_name] = distance # RotinaNet-6D
    return result, pt_cloud_paths, distances


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name, R1, R2, R3, R4, R5, R6, R7, R8, R9, T1, T2, T3 = row[:18]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name,R1, R2, R3, R4, R5, R6, R7, R8, R9, T1, T2, T3 \' '.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        # this probably works as is, but can probably be extended to handle all
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        # Bboxes
        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Rotaion Matrix 
        R1 = _parse(R1, float, 'line {}: malformed R1: {{}}'.format(line))
        R2 = _parse(R2, float, 'line {}: malformed R2: {{}}'.format(line))
        R3 = _parse(R3, float, 'line {}: malformed R3: {{}}'.format(line))
        R4 = _parse(R4, float, 'line {}: malformed R4: {{}}'.format(line))
        R5 = _parse(R5, float, 'line {}: malformed R5: {{}}'.format(line))
        R6 = _parse(R6, float, 'line {}: malformed R6: {{}}'.format(line))
        R7 = _parse(R7, float, 'line {}: malformed R7: {{}}'.format(line))
        R8 = _parse(R8, float, 'line {}: malformed R8: {{}}'.format(line))
        R9 = _parse(R9, float, 'line {}: malformed R9: {{}}'.format(line))

        # Translation-vector 
        T1 = _parse(T1, float, 'line {}: malformed T1: {{}}'.format(line))
        T2 = _parse(T2, float, 'line {}: malformed T2: {{}}'.format(line))
        T3 = _parse(T3, float, 'line {}: malformed T3: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # Check that rotation matrix is invalid, check for orthogonality  - RotinaNet-6D
        eps = 0.0001
        dot_prd = np.dot(np.array([R1,R2,R3]),np.array([R4,R5,R6]))
        if abs(dot_prd) > eps:
            raise ValueError('line {}: the row [R1,R2,R3] ([{},{},{}]) is not orthogonal with [R4,R5,R6]] ({},{},{}), dot product is {}'.format(line, R1,R2,R3, R4,R5,R6,dot_prd))
        dot_prd = np.dot(np.array([R1,R2,R3]),np.array([R7,R8,R9]))
        if abs(dot_prd) > eps:
            raise ValueError('line {}: the row [R1,R2,R3] ([{},{},{}]) is not orthogonal with [R7,R8,R9]] ({},{},{}), dot product is {}'.format(line, R1,R2,R3, R7,R8,R9,dot_prd))
        dot_prd = np.dot(np.array([R4,R5,R6]),np.array([R7,R8,R9]))
        if abs(dot_prd) > eps:
            raise ValueError('line {}: the row [R4,R5,R6] ([{},{},{}]) is not orthogonal with [R7,R8,R9]] ({},{},{}), dot product is {}'.format(line, R4,R5,R6, R7,R8,R9,dot_prd))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        # Rotation and translation are matrices/vectors from here on
        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name,
                                'rot': np.array([R1,R2,R3,R4,R5,R6,R7,R8,R9]), 'trans': np.array([T1,T2, (T3 - 0.6549)*1.0/0.4219])}) # Make a more elegant normalization 
    return result

def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        base_dir=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data  = {}
        self.diag_distances = {}
        self.base_dir    = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes, self.pt_cloud_paths, self.diag_distances = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.pt_clouds = {}
        for key, path in self.pt_cloud_paths.items():
            self.pt_clouds[key] = np.load(os.path.join(self.base_dir, path))

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def name_to_pt_cloud(self, name): # RotinaNet-6D
        """ Get the point cloud for a specific object.
        """
        return self.pt_clouds[name], self.diag_distances[name]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])


    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path        = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'rotations': np.empty((0, 9)), 'translations': np.empty((0,3))}

        for annot in self.image_data[path]:
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))
            annotations['rotations'] = np.concatenate((annotations['rotations'],[annot['rot']]))
            annotations['translations'] = np.concatenate((annotations['translations'],[annot['trans']]))
        return annotations

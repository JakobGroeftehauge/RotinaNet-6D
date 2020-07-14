# RotinaNet-6D
This project aims to develope a detector cabable of detecting and predicting the 6D pose of objects in RGB images. The structur of the detector will be based on the RetinaNet detector proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf). As a starting point the [Keras RetinaNet](https://github.com/fizyr/keras-retinanet) implementation provided by Fizyr will be used. 

## Benchmarking 
The detector will be benchmarked on the [LINEMOD](https://bop.felk.cvut.cz/datasets/) dataset.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
```
path/to/image.jpg,x1,y1,x2,y2,class_name, R00, R01, R02, R10, R11, R12, R20, R21, R22
```

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

# RotinaNet-6D
This project aims to develope a detector cabable of detecting and predicting the 6D pose of objects in RGB images. The structur of the detector will be based on the RetinaNet detector proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf). As a starting point the [Keras RetinaNet](https://github.com/fizyr/keras-retinanet) implementation provided by Fizyr will be used. 

## Installation 

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.

## Testing

## Benchmarking 
The detector will be benchmarked on the [LINEMOD](https://bop.felk.cvut.cz/datasets/) dataset.

The developed detectors performance on the dataset LINEMOD will be compared to the results introduced in the acrticle [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf) which is considered to be state of the art. 

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping, path to point_cloud_file and the length of the diagonal of the point cloud.  

### Annotations format
```
path/to/image.jpg,x1,y1,x2,y2,class_name, R00, R01, R02, R10, R11, R12, R20, R21, R22, T0, T1, T2
```

`(x1, y1), (x2, y2)` - Represents upper left and lower right image coordinates of the 2D bounding box enclosing the object
`(R00, R01, R02); (R10, R11, R12); (R20 R21, R22)`- Represents the rotations matrix
`(T0; T1; T2)` - Represents the translations vector 

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id,path/to/point_cloud.npy,diag_distance
```

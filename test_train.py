from keras_retinanet.bin.train import main

args = ['--steps', '79', '--epochs', '2', '--batch-size', '2', 'csv', 'data generation/train_data_ape.csv', 'data generation/classes.csv', '--val-annotations', 'data generation/rnd_val_data_ape.csv']

main(args)
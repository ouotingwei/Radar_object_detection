# Radar_Object_Detector

## training setep

1. active the conda env
```
conda activate yolo_test
```

2. label the training set
```
python3 label.py
```

3. pretest (double check before training)
```
python3 pretest.py
```

4. training
```
python3 training.py
```

5. png -> mp4
```
python3 png2mp4.py
```

6. predict 
```
python3 predict.py
```
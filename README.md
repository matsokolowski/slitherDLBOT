# slitherDLBOT
Simple slither.io CNN Deep Q bot written in python using Keras, Selenium, Numpy, Gstreamer, CV2, wmtool, openbox. The script records frames from the browser and your mouse movement to use it afterward for training. The whole project runs Xephyr with VirtualGL to make X11 stable enough for Gstreamer screen capture
## Recording samples: 
```
sh run-human-recording.sh
```

## Getting the best scores for latter training 
```
python gethebestrecorded.py 
```

## Training the model:

```
python train.py
```

## Starting bot:
```
sh run.sh
```

## Video: https://youtu.be/BoZk81DrpoA

#!/bin/bash
Xephyr -screen 1024x1024 :10& disown
DISPLAY=:10 openbox & disown 
sleep 1
DISPLAY=:10 xterm -e "while xdotool key Shift; do sleep 10 ; done"  & disown
DISPLAY=:10 lxterminal -e "vglrun python model.py; read" & disown

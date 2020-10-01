#! /usr/bin/python3
from ev3dev2.motor import Motor, SpeedDPS, SpeedPercent
from time import sleep
import os
os.system('setfont Lat15-TerminusBold14')


hand = Motor('A')
hand.position = 0
hand.on_to_position(0.05, 20 / 360 * hand.count_per_rot, block=False)
while True:
    sleep(1)
    print('\r', hand.position / hand.count_per_rot * 360, end='')

#! /usr/bin/python3
from ev3dev2.motor import Motor, SpeedDPS, SpeedPercent
from time import sleep
from threading import Thread
from math import pi
import os
os.system('setfont Lat15-TerminusBold14')


power_save = True
lines = [([0, 1, 2, 0.5], [0.3, 0.2, -0.1, 0])]


def main(lines):
    turner = Turner()
    turner.start(turns=False)
    hand = Hand(turner)
    hand.start()
    hand.goo()
    for line in lines:
        hand.move(0, 0.5, 0)
        for x, y in line:
            x = x / pi * 180
            y = y / 2 + 0.5
            hand.move(1, y, x)
    hand.move(0, 0, 0)
    hand.stop()
    return
    hand.move(0, 0.5, 0)
    hand.move(1, 0.5, 90)
    hand.move(1, 0.5, -30)
    hand.move(1, 0.5, 60)
    hand.move(1, 0.5, 0)
    hand.move(1, 0.5, -50)
    hand.move(0, 1, -80)
    hand.move(1, 1, -80)
    hand.move(1, 0.9, 80)
    hand.move(1, 1, -80)
    hand.move(0, 1, -80)
    hand.move(0, 0, 0)
    hand.stop()


class Hand(object):
    def __init__(self, turner, up_motor='B', hand_motor='A'):
        self.angle = turner.angle
        self.up = Motor(up_motor)
        self.hand = Motor(hand_motor)

        self.degrees = 21
        self.up_bounds = -100
        self.pwm_base = 0.1

        self.color_org = 0
        self.color_tgt = 0
        self.up_org = 0
        self.up_tgt = 0

    def speed(self):
        up = self.up.position / self.up_bounds
        tgt = self.color_tgt * self.degrees
        if up > 0.7:
            tgt += 3
        pos = self.hand.position / self.hand.count_per_rot * 360
        if abs(pos - tgt) < 5:
            if abs(pos - tgt) < 1:
                return 0.01, tgt
            return 0.1, tgt * 1.1
        else:
            return 5, tgt

    def push(self):
        vel, pos = self.speed()
        self.hand.on_to_position(
            vel, pos / 360 * self.hand.count_per_rot, block=False)

    def start(self):
        self.hand.position = 0
        self.up.position = 0
        self.angle.position = 0
        self.wait()

    def wait(self):
        self.up.wait_until_not_moving()
        self.angle.wait_until_not_moving()
        self.hand.wait_until_not_moving()

    def goo(self):
        def fun():
            while True:
                self.push()
                sleep(0.05)

        self.thread = Thread(target=fun)
        self.thread.start()

    def stop(self):
        self.thread._stop()

    def move(self, color, up, angle, lift=False):
        self.up_org = self.up_tgt
        self.up_tgt = up
        self.color_org = self.color_tgt
        self.color_tgt = color
        self.up.on_to_position(10, up * self.up_bounds, block=False)
        self.angle.on_to_position(
            10, angle / 360 * self.angle.count_per_rot, block=False)
        self.wait()


class Turner(object):
    def __init__(self, arm_motor='C', angle_motor='D'):
        self.arm = Motor(arm_motor)
        self.angle = Motor(angle_motor)

    def start(self, turns=True):
        if power_save:
            print("brrr")
        else:
            print("it's  time")
            self.arm.on(30)
            sleep(5)
            self.arm.on(15)
            if turns:
                for _ in range(10):
                    self.angle.on_for_seconds(-50, 0.2)
                    self.angle.on_for_seconds(50, 0.2)

        self.angle.on(10)


if __name__ == "__main__":
    main(lines)

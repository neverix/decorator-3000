#! /usr/bin/python3
from ev3dev2.motor import Motor, SpeedDPS, SpeedPercent
from time import sleep
from threading import Thread


def main():
    turner = Turner()
    turner.start()
    hand = Hand(turner)
    hand.start()
    hand.move(-1, 1, 0)
    sleep(1)
    hand.move(1, 0.5, 90)
    sleep(1)
    hand.move(-1, 0.3, 45)
    sleep(1)
    hand.move(0, 0, 0)
    sleep(2)


class Hand(object):
    def __init__(self, turner, up_motor='B', hand_motor='A'):
        self.angle = turner.angle
        self.up = Motor(up_motor)
        self.hand = Motor(hand_motor)
        self.degrees = 30
        self.rot = self.hand.count_per_rot / 360 * self.degrees
        self.up_bounds = -100

    def start(self):
        self.hand.position = 0
        self.up.position = 0
        self.angle.position = 0
        sleep(1)

    def move(self, color, up, angle):
        self.hand.stop()
        self.hand.on_to_position(
            10, self.rot * color * -1, block=False)
        self.up.on_to_position(10, up * self.up_bounds,
                               block=False)
        self.angle.on_to_position(10, angle / 360 * self.angle.count_per_rot)


class Turner(object):
    def __init__(self, arm_motor='C', angle_motor='D'):
        self.arm = Motor(arm_motor)
        self.angle = Motor(angle_motor)

    def start(self):
        self.arm.on(15)
        sleep(10)
        for _ in range(10):
            self.angle.on_for_seconds(50, 0.2)
            self.angle.on_for_seconds(-50, 0.2)
        self.angle.on(10)


if __name__ == "__main__":
    main()

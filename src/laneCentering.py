import numpy as np
from scipy import signal as f
import vgamepad as vg
import time

Kp = 0.006
Ki = 0.00175
Kd = 0.0
t = time.time()
staticSetpoint = 0
clamp = [-0.45, 0.45]

deviationBuffer = []
lowpass = f.butter(3, 0.75, output='sos')  # 0.85

gamepad = vg.VX360Gamepad()
gamepad.update()


def laneCentering(laneBoundaries, target, v):
    midpoints = []

    for set in laneBoundaries:
        if not set[0] == 0 and not set[1] == 0:
            midpoints.append(set[2])

    # Optimization target
    halfway = len(midpoints) // 2
    deviation = np.mean(midpoints[0:halfway]) - target  # target of 400, midscreen

    if not np.isnan(deviation):

        deviationBuffer.insert(0, deviation)
        if (len(deviationBuffer) > 20):
            deviationBuffer.pop()

        filteredBuffer = f.sosfilt(lowpass, deviationBuffer)

        corrector = pid(filteredBuffer[0], staticSetpoint)

        gamepad.left_joystick_float(x_value_float=corrector, y_value_float=0.0)

        gamepad.update()

    else:
        gamepad.reset()
        gamepad.update()


def pid(i, setpoint):
    global t

    error = setpoint - i
    dt = time.time() - t
    t = time.time()

    proportional = cV(Kp * error)
    integral = cV(Ki * error * dt)
    derivative = cV(Kd * error / dt)
    output = cV(proportional + integral + derivative)

    return output


def cV(val):
    global clamp

    if val > clamp[1]:
        val = clamp[1]
    elif val < clamp[0]:
        val = clamp[0]

    return val

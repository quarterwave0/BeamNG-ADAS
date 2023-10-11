from simple_pid import PID
import numpy as np
from scipy import signal as f
import vgamepad as vg

Kp = 0.006
Ki = 0.00175
Kd = 0.0

p = PID(Kp, Ki, Kd, setpoint=0)
p.output_limits = (-0.45, 0.45)

deviationBuffer = []
lowpass = f.butter(3, 0.75, output='sos') #0.85

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

        corrector = p(v)
        pO, iO, dO = p.components
        print(corrector, filteredBuffer[0], pO, iO, dO)

        gamepad.left_joystick_float(x_value_float=corrector, y_value_float=0.0)

        gamepad.update()
        vn = filteredBuffer[0]

    else:
        vn = v
        gamepad.reset()
        gamepad.update()

    return vn
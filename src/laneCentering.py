from simple_pid import PID
import numpy as np
from scipy import signal as f
import vgamepad as vg

#Kp = 0.0057
Kp = 0.006
Ki = 0.0006
Kd = 0.00255

p = PID(Kp, Ki, Kd, setpoint=0)
p.output_limits = (-0.35, 0.35)

deviationBuffer = []
lowpass = f.butter(3, 0.85, output='sos')

gamepad = vg.VX360Gamepad()
gamepad.update()

def laneCentering(laneBoundaries, target, v):

    midpoints = []

    for set in laneBoundaries:
        if not set[0] == 0 and not set[1] == 0:
            midpoints.append(set[2])

    # Optimization target
    deviation = np.mean(midpoints) - target  # target of 400

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

        # if(deviation<-15):
        #     print(deviation, "deviating left")
        #     gamepad.left_joystick_float(x_value_float=0.3, y_value_float=0.0)
        #     gamepad.update()
        # elif (deviation>15):
        #     print(deviation, "deviating right")
        #     gamepad.left_joystick_float(x_value_float=-0.3, y_value_float=0.0)
        #     gamepad.update()

    else:
        vn = v
        gamepad.reset()
        gamepad.update()

    return vn
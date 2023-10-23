# BeamNG.Drive ADAS
### (Or, the making of a car that drives itself in a game about driving)

This project is a pretty simple OpenCV based implementation of ADAS in the game [BeamNG.drive](https://www.beamng.com/game/).

![A functional diagram of the ADAS system](./docs/ADAS.png)


# Libaries
- [NumPy](https://numpy.org/) is great!
- And so is [SciPy](https://scipy.org/)! I use it for uni-variate spline fitting, as well as for the digital filters.
- [vgamepad](https://github.com/yannbouteiller/vgamepad) is the tool used for all steering commands. Works really well!
- How could I forget about the centerpiece, [OpenCV](https://opencv.org/), which I use for image capture, some manipulation tasks, drawing, and display
# References
- The wonderful code from the [Udacity Self-Driving Car Engineer Nanodegree](https://github.com/ndrplz/self-driving-car/tree/master) was a huge help in getting oriented on lane detection.
- M. Aly, “Real time detection of Lane markers in urban streets,” 2008 IEEE Intelligent Vehicles Symposium, Jun. 2008. doi:10.1109/ivs.2008.4621152
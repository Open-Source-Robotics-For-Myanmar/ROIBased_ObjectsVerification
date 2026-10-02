"""
kalman_tracker.py
2D Constant-Velocity Kalman Filter for target center (cx, cy).

State   : [x, y, vx, vy]^T
Measure : [x, y]^T
"""

import numpy as np
import cv2


class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=5e-1):
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition  F
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix  H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Noise covariances
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)

        self.initialized = False

    def init(self, x, y):
        """Initialize state with first measurement, zero velocity."""
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kf.statePre  = self.kf.statePost.copy()
        self.initialized  = True

    def predict(self):
        """Predicted state (uses constant velocity)."""
        return self.kf.predict()

    def correct(self, x, y):
        """Update with measurement; returns corrected state."""
        z = np.array([[np.float32(x)], [np.float32(y)]])
        return self.kf.correct(z)
package cc.briem.swift;

import java.io.ByteArrayInputStream;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BlockingQueue;

import cc.briem.swift.cv.models.HandLandmarks;
import cc.briem.swift.imu.IMUController;
import cc.briem.swift.imu.ImuPacket;
import javafx.geometry.Point3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.cv.models.LandmarkResult;
import cc.briem.swift.network.models.Frame;

import javafx.application.Platform;
import javafx.scene.image.Image;

public class AnalysisThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(AnalysisThread.class);

    private final BlockingQueue<Frame> analyzedFrames;
    private final BlockingQueue<LandmarkResult> landmarkResults;
    private final IMUController imuController;

    // Moving Average Config
    int windowSize = 5;
    HandLandmarks[] movingAvgBuffer = new HandLandmarks[windowSize];
    int index = 0;
    int count = 0;
    HandLandmarks runningSum = null;

    // Low Pass Filter Config
    HandLandmarks previousFilteredFrame = null;
    float alpha = 0.8f;

    public AnalysisThread(BlockingQueue<Frame> analyzedFrames, BlockingQueue<LandmarkResult> landmarkResults, IMUController imuController) {
        this.analyzedFrames = analyzedFrames;
        this.landmarkResults = landmarkResults;
        this.imuController = imuController;
    }

    @Override
    public void run() {
        logger.info("AnalysisThread starting...");

        while (!Thread.currentThread().isInterrupted()) {
            try {
                // Take frame
                Frame frame = analyzedFrames.take();
                analyzedFrames.clear(); // drop backlog, always display the latest frame

                Image image = new Image(new ByteArrayInputStream(frame.getJpegData()));

                // Get IMU data at frame time
                Instant now = Instant.now();
                ImuPacket.Acceleration accelPacket = imuController.getImuAt(now, ImuPacket.Acceleration.class).get();
                ImuPacket.Angle anglePacket = imuController.getImuAt(now, ImuPacket.Angle.class).get();
                ImuPacket.Gyro gyroPacket = imuController.getImuAt(now, ImuPacket.Gyro.class).get();
                ImuPacket[] imuPackets = new ImuPacket[3];
                imuPackets[0] = accelPacket;
                imuPackets[1] = anglePacket;
                imuPackets[2] = gyroPacket;

                // Predict landmarks from frame
                LandmarkResult landmarkResult = landmarkResults.poll();
                landmarkResults.clear(); // keep in sync with frame backlog drop above
                if(landmarkResult == null) continue;
                List<HandLandmarks> hands = landmarkResult.getHands();

                // -- PROCESS ENDS HERE WHEN NO FRAME --

                // Filter chain
                if (!hands.isEmpty()) {
                    // Moving Average of Landmarks
                    //HandLandmarks moving = movingAverage(hands.getFirst());

                    // Low Pass Filter
                    HandLandmarks filtered = lowPassFilter(hands.getFirst());

                    // Set Results
                    landmarkResult.setFirstHandLandmarks(filtered);
                    hands.set(0, filtered);
                } else {
                    resetMovingAvgBuffer();
                }

                // Calculate palm normals
                Point3D[] imuNormal = new Point3D[2];
                imuNormal[0] = imuNormal(anglePacket);
                imuNormal[1] = imuNormal[0]; // Only 1 sensor for now
                Point3D[] geometricNormal = new Point3D[2];
                geometricNormal[0] = new Point3D(0, 0, 0); // ensures not null
                geometricNormal[1] = new Point3D(0, 0, 0); // ensures not null
                if(!hands.isEmpty()) {
                    geometricNormal[0] = geometricNormal(hands.getFirst());
                    if(hands.size() > 1) {
                        geometricNormal[1] = geometricNormal(hands.getLast());
                    }
                }

                // Calculate angle between vectors (1 hand only)
                double vectorAngle = angleBetween(imuNormal[0], geometricNormal[0]);
                TrackingState state = getTrackingState(geometricNormal[0], imuNormal[0]);

                // Calculate moving Average of HandLandmarks
                if (!hands.isEmpty()) {
                    HandLandmarks moving = movingAverage(hands.getFirst());
                    landmarkResult.setFirstHandLandmarks(moving);
                } else {
                    resetMovingAvgBuffer();
                }


                Platform.runLater(() -> DisplayApp.render(image, landmarkResult, imuNormal, geometricNormal, state));

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.info("AnalysisThread interrupted, shutting down.");
            }
        }
    }

    private HandLandmarks kalmanFuse(ImuPacket[] imuPackets, HandLandmarks handLandmarks) {
        /** What kalman does:
         *
         * - CV Frame (when visible) is absolute truth
         * - Between frames, use the IMU data to predict new landmark position
         * - When ext frame arrives, update to truth again
         *
         * Unscented or Extended Kalman Filter for 3D movement (nonlinear)
         *
         * EKF vs. UKF = Calculus vs. Statistics (sampling)
         */
        return null;
    }

    private HandLandmarks movingAverage(HandLandmarks newFrame) {
        if(newFrame == null) {
            return null;
        }

        if(runningSum == null) {
            movingAvgBuffer[index] = newFrame;
            runningSum = newFrame;
            count = 1;
            index = (index + 1) % windowSize;
            return newFrame;
        }

        if(count == windowSize) {
            HandLandmarks oldestFrame = movingAvgBuffer[index];
            runningSum = runningSum.subtract(oldestFrame);
        } else {
            count++;
        }

        runningSum = runningSum.add(newFrame);
        movingAvgBuffer[index] = newFrame;

        index = (index + 1) % windowSize;

        return runningSum.divide(count);
    }

    private void resetMovingAvgBuffer() {
        Arrays.fill(movingAvgBuffer, null);
        index = 0;
        count = 0;
        runningSum = null;
    }

    private HandLandmarks lowPassFilter(HandLandmarks newFrame) {
        if(newFrame == null) {
            return null;
        }

        if(previousFilteredFrame == null) {
            previousFilteredFrame = newFrame;
            return newFrame;
        }

        // S_t = alpha * Y_t + (1 - alpha) * S_(t-1) (S: Smoothed Frame at t, Y: incoming Frame at t, alpha: smoothing factor (exp)
        HandLandmarks scaledNew = newFrame.multiply(alpha);
        HandLandmarks scaledPrev = previousFilteredFrame.multiply(1.0f - alpha);

        HandLandmarks smoothedFrame = scaledNew.add(scaledPrev);

        previousFilteredFrame = smoothedFrame;

        return smoothedFrame;
    }

    // -------------------------------------------------------------------------
    // IMU arrow
    // -------------------------------------------------------------------------

    /**
     * Fixed mounting correction matrix M.
     *
     * <p>Encodes the physical orientation of the sensor relative to the world frame.
     * Each column is where the corresponding sensor basis vector points in world coordinates:
     * <pre>
     *   Sensor +X → world +Z  (toward camera)
     *   Sensor +Y → world +X  (camera's right)
     *   Sensor +Z → world -Y  (down)
     * </pre>
     *
     * <pre>
     *       sX  sY  sZ
     *   M = [ 0   1   0 ]   world X
     *       [ 0   0  -1 ]   world Y
     *       [ 1   0   0 ]   world Z
     * </pre>
     */
    public static final double[][] M = {
            { 0,  1,  0 },
            { 0,  0, -1 },
            { 1,  0,  0 },
    };

    /**
     * Computes the IMU sensor's normal vector in world coordinates.
     *
     * <p>Applies {@code n_world = M * R(roll,pitch,yaw) * (0,0,-1)}, where:
     * <ul>
     *   <li>{@code (0,0,-1)} is the sensor's outward normal in its own frame (-Z face)</li>
     *   <li>{@code R} is the ZYX Euler rotation reported by the WIT sensor</li>
     *   <li>{@code M} is the fixed mounting correction matrix above</li>
     * </ul>
     *
     * @param angle IMU angle packet (roll, pitch, yaw in degrees)
     * @return unit normal vector in world coordinates
     */
    public static Point3D imuNormal(ImuPacket.Angle angle) {
        double r = Math.toRadians(angle.roll());
        double p = Math.toRadians(angle.pitch());
        double y = Math.toRadians(angle.yaw());

        double sr = Math.sin(r), cr = Math.cos(r);
        double sp = Math.sin(p), cp = Math.cos(p);
        double sy = Math.sin(y), cy = Math.cos(y);

        // R * (0,0,-1) = negated third column of Rz(y)*Ry(p)*Rx(r)
        double vx = -(cy * sp * cr + sy * sr);
        double vy = -(sy * sp * cr - cy * sr);
        double vz = -(cp * cr);

        // n_world = M * v
        double nx = M[0][0] * vx + M[0][1] * vy + M[0][2] * vz;
        double ny = M[1][0] * vx + M[1][1] * vy + M[1][2] * vz;
        double nz = M[2][0] * vx + M[2][1] * vy + M[2][2] * vz;

        return new Point3D(nx, ny, nz);
    }

    /**
     * Computes the palms normal vector in world coordinates from computed landmarks.
     *
     * @param landmarks HandLandmarks element
     * @return unit normal vector in world coordinates
     */
    public static Point3D geometricNormal(HandLandmarks landmarks) {
        Point3D point1 = landmarks.points.get(0);
        Point3D point2 = landmarks.points.get(1);
        Point3D point3 = landmarks.points.get(2);

        // Compute two edge vectors of the triangle
        double ux = point2.getX() - point1.getX();
        double uy = point2.getY() - point1.getY();
        double uz = point2.getZ() - point1.getZ();

        double vx = point3.getX() - point1.getX();
        double vy = point3.getY() - point1.getY();
        double vz = point3.getZ() - point1.getZ();

        // Cross product u x v gives a vector perpendicular to the plane
        double nx = uy * vz - uz * vy;
        double ny = uz * vx - ux * vz;
        double nz = ux * vy - uy * vx;

        // Normalize the resulting vector
        double length = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (length > 0) {
            nx /= length;
            ny /= length;
            nz /= length;
        }

        // --- CENTROID FLIP CHECK ---
        // 1. Calculate the centroid (average position) of all landmarks
        double centroidX = 0, centroidY = 0, centroidZ = 0;
        int totalPoints = landmarks.points.size();

        for (Point3D p : landmarks.points) {
            centroidX += p.getX();
            centroidY += p.getY();
            centroidZ += p.getZ();
        }
        centroidX /= totalPoints;
        centroidY /= totalPoints;
        centroidZ /= totalPoints;

        // 2. Create a vector from point1 (on the plane) to the centroid
        double toCentroidX = centroidX - point1.getX();
        double toCentroidY = centroidY - point1.getY();
        double toCentroidZ = centroidZ - point1.getZ();

        // 3. Compute dot product between the normal and the centroid vector
        double dotProduct = (nx * toCentroidX) + (ny * toCentroidY) + (nz * toCentroidZ);

        // 4. If dot product is positive, normal points toward the hand interior. Flip it!
        if (dotProduct > 0) {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }

        return new Point3D(nx, ny, nz).multiply(-1);
    }

    public static TrackingState getTrackingState(Point3D geometricNormal, Point3D imuNormal) {
        if(geometricNormal == null || ( geometricNormal.getX() == 0.0 && geometricNormal.getY() == 0.0 && geometricNormal.getZ() == 0.0)) {
            return TrackingState.OCCLUDED;
        }

        double angle = angleBetween(geometricNormal, imuNormal) % 180;

        if (angle < 10) {
            return TrackingState.TRACKING;
        } else {
            return TrackingState.MISMATCH;
        }
    }

    private static double angleBetween(Point3D vector1, Point3D vector2) {
        return vector1.angle(vector2);
    }
}


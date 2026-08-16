package cc.briem.swift;

import java.io.ByteArrayInputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

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

    // Kalman fusion state (wrist only — see kalmanFuse)
    private UKF.WristFilter wristFilter;
    private Instant lastPredictTime;
    private List<Point3D> referenceOffsets;   // points 0..20 minus wrist, metric world frame, from the last confident CV frame
    private ImuPacket.Angle referenceAngle;   // IMU orientation captured alongside referenceOffsets
    private double lastDepthScale;            // pixel->metric depth scale from the last confident CV frame, reused during occlusion
    private List<Point3D> lastWorldPoints;
    private double lastHandednessScore;
    private double lastPresenceScore;

    private static final long PREDICT_TICK_MS = 10;        // ~100Hz predict/render tick, decoupled from actual IMU/CV rates
    private static final double PRESENCE_THRESHOLD = 0.5;  // gate for treating a CV frame as "confident"
    private static final double GRAVITY_MPS2 = 9.80665;

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
                // Tick at a fixed, high rate so IMU-driven prediction isn't gated on CV frame
                // arrival (analyzedFrames.take() would only wake up at CV cadence, ~30Hz).
                Frame frame = analyzedFrames.poll(PREDICT_TICK_MS, TimeUnit.MILLISECONDS);
                if (frame != null) {
                    analyzedFrames.clear(); // drop backlog, always display the latest frame
                }

                // Get IMU data at tick time
                Instant now = Instant.now();
                Optional<ImuPacket.Acceleration> accelPacket = imuController.getImuAt(now, ImuPacket.Acceleration.class);
                Optional<ImuPacket.Angle> anglePacketOpt = imuController.getImuAt(now, ImuPacket.Angle.class);
                Optional<ImuPacket.Gyro> gyroPacket = imuController.getImuAt(now, ImuPacket.Gyro.class);
                if (accelPacket.isEmpty() || anglePacketOpt.isEmpty() || gyroPacket.isEmpty()) {
                    continue; // IMU hasn't produced its first packet(s) of every type yet
                }
                ImuPacket.Angle anglePacket = anglePacketOpt.get();
                ImuPacket[] imuPackets = new ImuPacket[3];
                imuPackets[0] = accelPacket.get();
                imuPackets[1] = anglePacket;
                imuPackets[2] = gyroPacket.get();

                // Predict landmarks from frame, if one arrived this tick
                LandmarkResult landmarkResult = null;
                HandLandmarks rawHand = null;
                if (frame != null) {
                    landmarkResult = landmarkResults.poll();
                    landmarkResults.clear(); // keep in sync with frame backlog drop above
                    if (landmarkResult != null && !landmarkResult.getHands().isEmpty()) {
                        rawHand = landmarkResult.getHands().getFirst();
                    }
                }

                // Kalman-fuse the wrist: CV is absolute truth when confident, IMU carries the
                // estimate between frames / through occlusion. Runs every tick (predict), and
                // additionally corrects (update) whenever a confident CV frame is present.
                // Replaces the old lowPass/movingAverage chain (kept below, unused, for reference).
                //HandLandmarks fused = kalmanFuse(imuPackets, rawHand);
                HandLandmarks fused = rawHand;

                if (frame == null || landmarkResult == null) {
                    continue; // no new image this tick — nothing to (re)render
                }

                List<HandLandmarks> hands = landmarkResult.getHands();
                if (fused != null) {
                    if (hands.isEmpty()) {
                        hands.add(fused);
                    } else {
                        landmarkResult.setFirstHandLandmarks(fused);
                    }
                }

                Image image = new Image(new ByteArrayInputStream(frame.getJpegData()));

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

                final LandmarkResult resultForRender = landmarkResult;
                Platform.runLater(() -> DisplayApp.render(image, resultForRender, imuNormal, geometricNormal, state));

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
         *
         * UKF: Higher accuracy during intense movements, easier to implement <-- Choose this
         *
         * Only evaluate / modify Landmark 0 (Wrist) : Too much load
         * --> IMU angular velocity does not matter then ( can just be applied )
         * ==> 6 Dimensional Data Points
         */

        ImuPacket.Acceleration accel = (ImuPacket.Acceleration) imuPackets[0];
        ImuPacket.Angle angle = (ImuPacket.Angle) imuPackets[1];

        Instant now = Instant.now();

        Point3D wristWorld = null;
        double depthScale = 0.0;
        if (handLandmarks != null) {
            depthScale = computeDepthScale(handLandmarks);
            if (depthScale > 0.0) {
                wristWorld = computeWristWorldPosition(handLandmarks, depthScale);
            }
        }
        boolean confident = wristWorld != null && handLandmarks.presenceScore >= PRESENCE_THRESHOLD;

        if (wristFilter == null) {
            if (!confident) {
                return null; // nothing to bootstrap the filter from yet
            }
            wristFilter = new UKF.WristFilter(wristWorld);
            captureReferenceShape(handLandmarks, angle, depthScale);
            lastPredictTime = now;
            return reprojectHand(wristFilter.getPosition(), angle);
        }

        double dt = (lastPredictTime == null) ? 0.0 : Duration.between(lastPredictTime, now).toNanos() / 1_000_000_000.0;
        lastPredictTime = now;
        if (dt > 0.0) {
            wristFilter.predict(dt, sensorAccelerationMps2(accel), worldRotationMatrix(angle));
        }

        if (confident) {
            wristFilter.update(wristWorld);
            captureReferenceShape(handLandmarks, angle, depthScale);
        }

        return reprojectHand(wristFilter.getPosition(), angle);
    }

    /**
     * Standard pinhole back-projection of the wrist's own pixel position, using {@code depthScale}
     * (from {@link #computeDepthScale}) as its absolute camera-to-wrist distance in metres.
     *
     * <p>Deliberately does NOT use {@link UKF#toWorldCoordinates} here: that helper multiplies each
     * landmark's own {@code z} by depthScale, but MediaPipe defines landmark 0's z relative to
     * itself (≈0, since the wrist is the depth *reference*) — so its output for the wrist is always
     * a noise-dominated point near the origin, regardless of where the wrist actually is. Using the
     * palm-width-derived depthScale directly as the wrist's absolute Z avoids that degeneracy.
     */
    private static Point3D computeWristWorldPosition(HandLandmarks pixelHand, double depthScale) {
        Point3D wristPixel = pixelHand.points.get(0);
        double zCam = depthScale;
        double xCam = (wristPixel.getX() - UKF.CameraIntrinsics.MBP.cx()) / UKF.CameraIntrinsics.MBP.fx() * zCam;
        double yCam = (wristPixel.getY() - UKF.CameraIntrinsics.MBP.cy()) / UKF.CameraIntrinsics.MBP.fy() * zCam;
        return new Point3D(xCam, -yCam, -zCam);
    }

    /**
     * Caches the wrist-relative offsets + IMU orientation + depth scale of a confident CV frame.
     *
     * <p>Offsets come from {@code handLandmarks.worldPoints} — MediaPipe's own native metric hand
     * pose (already correct and already in the X-right/Y-up/Z-toward-camera convention used
     * throughout this class) — rather than from {@link UKF#toWorldCoordinates}, whose per-point
     * math has the same self-relative-z issue described in {@link #computeWristWorldPosition}.
     */
    private void captureReferenceShape(HandLandmarks handLandmarks, ImuPacket.Angle angle, double depthScale) {
        Point3D wristOrigin = handLandmarks.worldPoints.get(0);
        List<Point3D> offsets = new ArrayList<>(handLandmarks.worldPoints.size());
        for (Point3D p : handLandmarks.worldPoints) {
            offsets.add(p.subtract(wristOrigin));
        }
        this.referenceOffsets = offsets;
        this.referenceAngle = angle;
        this.lastDepthScale = depthScale;
        this.lastWorldPoints = handLandmarks.worldPoints;
        this.lastHandednessScore = handLandmarks.handednessScore;
        this.lastPresenceScore = handLandmarks.presenceScore;
    }

    /**
     * Rigidly rotates the cached reference hand shape by however much the IMU orientation has
     * changed since it was captured, anchors it at the fused wrist position, and projects the
     * result back to image pixel coordinates (the frame the rest of the pipeline — DisplayApp,
     * geometricNormal — expects {@code HandLandmarks.points} to be in).
     */
    private HandLandmarks reprojectHand(Point3D fusedWristWorld, ImuPacket.Angle angle) {
        if (referenceOffsets == null) {
            return null;
        }

        double[][] delta = multiply3x3(worldRotationMatrix(angle), transpose3x3(worldRotationMatrix(referenceAngle)));

        List<Point3D> pixelPoints = new ArrayList<>(referenceOffsets.size());
        for (Point3D offset : referenceOffsets) {
            Point3D worldPoint = fusedWristWorld.add(applyMatrix(delta, offset));
            pixelPoints.add(toPixelSpace(worldPoint, lastDepthScale));
        }

        return new HandLandmarks(pixelPoints, lastWorldPoints, lastHandednessScore, lastPresenceScore);
    }

    /**
     * Raw accelerometer packet converted to sensor-frame specific force (m/s^2). Rotation into
     * world frame, gravity compensation, and accelerometer-bias cancellation all now happen
     * inside {@link UKF.WristFilter}'s process model instead of here, since the bias correction
     * must be applied per-sigma-point (using that sigma point's own bias hypothesis) for the
     * filter to be able to observe and estimate the bias at all.
     *
     * <p>The raw vector is negated: on this hardware, the at-rest reading maps (via {@code M})
     * to the opposite of what was first assumed, which left gravity doubling instead of
     * cancelling (~2g of constant phantom downward acceleration) — confirmed by on-device testing
     * showing a fast, consistent downward drift between CV corrections.
     */
    private static Point3D sensorAccelerationMps2(ImuPacket.Acceleration accel) {
        return new Point3D(accel.ax(), accel.ay(), accel.az()).multiply(-GRAVITY_MPS2);
    }

    /** Mirrors UKF.toWorldCoordinates' internal depth-scale calc (landmarks 0 and 9 = its PALM_REF_A/B). */
    private static double computeDepthScale(HandLandmarks pixelHand) {
        Point3D refA = pixelHand.points.get(0);
        Point3D refB = pixelHand.points.get(9);
        double dx = refB.getX() - refA.getX();
        double dy = refB.getY() - refA.getY();
        double palmWidthPx = Math.sqrt(dx * dx + dy * dy);
        if (palmWidthPx == 0.0) return 0.0;
        return (UKF.DEFAULT_PALM_WIDTH_METRES * UKF.CameraIntrinsics.MBP.fx()) / palmWidthPx;
    }

    /** Inverse of {@link UKF#toWorldCoordinates}: projects a metric world point back to image pixels. */
    private static Point3D toPixelSpace(Point3D worldPoint, double depthScale) {
        double xCam = worldPoint.getX();
        double yCam = -worldPoint.getY();
        double zCam = -worldPoint.getZ();
        if (depthScale <= 0.0 || zCam == 0.0) {
            return new Point3D(UKF.CameraIntrinsics.MBP.cx(), UKF.CameraIntrinsics.MBP.cy(), 0);
        }
        double u = xCam / zCam * UKF.CameraIntrinsics.MBP.fx() + UKF.CameraIntrinsics.MBP.cx();
        double v = yCam / zCam * UKF.CameraIntrinsics.MBP.fy() + UKF.CameraIntrinsics.MBP.cy();
        double zPixel = zCam / depthScale;
        return new Point3D(u, v, zPixel);
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
        return rotateSensorToWorld(new Point3D(0, 0, -1), angle);
    }

    /**
     * Full sensor-to-world rotation matrix {@code M * R(roll,pitch,yaw)}, generalizing the
     * {@code n_world = M * R * (0,0,-1)} formula above to arbitrary sensor-frame vectors
     * (used both for {@link #imuNormal} and to rotate IMU acceleration / the rest of the
     * hand's landmarks into world coordinates in {@code kalmanFuse}).
     */
    private static double[][] worldRotationMatrix(ImuPacket.Angle angle) {
        double r = Math.toRadians(angle.roll());
        double p = Math.toRadians(angle.pitch());
        double y = Math.toRadians(angle.yaw());

        double sr = Math.sin(r), cr = Math.cos(r);
        double sp = Math.sin(p), cp = Math.cos(p);
        double sy = Math.sin(y), cy = Math.cos(y);

        // R = Rz(yaw) * Ry(pitch) * Rx(roll)
        double[][] rot = {
                { cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr },
                { sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr },
                { -sp,     cp * sr,                cp * cr                }
        };

        return multiply3x3(M, rot);
    }

    private static Point3D rotateSensorToWorld(Point3D sensorVec, ImuPacket.Angle angle) {
        return applyMatrix(worldRotationMatrix(angle), sensorVec);
    }

    private static Point3D applyMatrix(double[][] mat, Point3D v) {
        double x = v.getX(), y = v.getY(), z = v.getZ();
        return new Point3D(
                mat[0][0] * x + mat[0][1] * y + mat[0][2] * z,
                mat[1][0] * x + mat[1][1] * y + mat[1][2] * z,
                mat[2][0] * x + mat[2][1] * y + mat[2][2] * z
        );
    }

    private static double[][] multiply3x3(double[][] a, double[][] b) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double sum = 0;
                for (int k = 0; k < 3; k++) sum += a[i][k] * b[k][j];
                out[i][j] = sum;
            }
        }
        return out;
    }

    private static double[][] transpose3x3(double[][] a) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                out[j][i] = a[i][j];
            }
        }
        return out;
    }

    /**
     * Computes the palms normal vector in world coordinates from computed landmarks.
     *
     * @param landmarks HandLandmarks element
     * @return unit normal vector in world coordinates
     */
    public static Point3D geometricNormal(HandLandmarks landmarks) {
        Point3D point1 = landmarks.absolutePoints.get(0);
        Point3D point2 = landmarks.absolutePoints.get(5);
        Point3D point3 = landmarks.absolutePoints.get(17);

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


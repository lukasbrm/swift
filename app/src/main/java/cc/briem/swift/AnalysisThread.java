package cc.briem.swift;

import java.io.ByteArrayInputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
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

    // Kalman fusion state (wrist only — see kalmanFuse)
    private UKF.WristFilter wristFilter;
    private Instant lastPredictTime;
    private List<Point3D> referenceOffsets;         // landmarks 0..20 minus wrist, IMU/world frame, from the last confident CV frame
    private ImuPacket.Quaternion referenceOrientation; // IMU orientation captured alongside referenceOffsets
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
                Optional<ImuPacket.Quaternion> quaternionPacketOpt = imuController.getImuAt(now, ImuPacket.Quaternion.class);
                Optional<ImuPacket.Gyro> gyroPacket = imuController.getImuAt(now, ImuPacket.Gyro.class);
                if (accelPacket.isEmpty() || quaternionPacketOpt.isEmpty() || gyroPacket.isEmpty()) {
                    continue; // IMU hasn't produced its first packet(s) of every type yet
                }
                ImuPacket.Quaternion quaternionPacket = quaternionPacketOpt.get();
                ImuPacket[] imuPackets = new ImuPacket[3];
                imuPackets[0] = accelPacket.get();
                imuPackets[1] = quaternionPacket;
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
                HandLandmarks fused = kalmanFuse(imuPackets, rawHand);

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
                imuNormal[0] = imuNormal(quaternionPacket);
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
        ImuPacket.Quaternion orientation = (ImuPacket.Quaternion) imuPackets[1];

        Instant now = Instant.now();

        List<Point3D> absolutePoints = (handLandmarks != null) ? handLandmarks.absolutePoints : null;
        boolean confident = absolutePoints != null && !absolutePoints.isEmpty()
                && handLandmarks.presenceScore >= PRESENCE_THRESHOLD;
        Point3D wristWorld = confident ? flipYZ(absolutePoints.get(0)) : null;

        if (wristFilter == null) {
            if (!confident) {
                return null; // nothing to bootstrap the filter from yet
            }
            wristFilter = new UKF.WristFilter(wristWorld);
            captureReferenceShape(absolutePoints, orientation, handLandmarks);
            lastPredictTime = now;
            return reprojectHand(wristFilter.getPosition(), orientation);
        }

        double dt = (lastPredictTime == null) ? 0.0 : Duration.between(lastPredictTime, now).toNanos() / 1_000_000_000.0;
        lastPredictTime = now;
        if (dt > 0.0) {
            wristFilter.predict(dt, sensorAccelerationMps2(accel), worldRotationMatrix(orientation));
        }

        if (confident) {
            wristFilter.update(wristWorld);
            captureReferenceShape(absolutePoints, orientation, handLandmarks);
        }

        return reprojectHand(wristFilter.getPosition(), orientation);
    }

    /**
     * Caches the wrist-relative offsets + IMU orientation of a confident CV frame.
     *
     * <p>Offsets come from {@code handLandmarks.absolutePoints} — the hand's real absolute position
     * in camera space (solvePnP, see {@link HandLandmarks#getAbsoluteWorldPoints}) — converted to
     * the IMU/world convention via {@link #flipYZ} so the rigid rotation in {@link #reprojectHand}
     * lines up with {@link #worldRotationMatrix}.
     */
    private void captureReferenceShape(List<Point3D> absolutePoints, ImuPacket.Quaternion orientation, HandLandmarks handLandmarks) {
        Point3D wristWorld = flipYZ(absolutePoints.get(0));
        List<Point3D> offsets = new ArrayList<>(absolutePoints.size());
        for (Point3D p : absolutePoints) {
            offsets.add(flipYZ(p).subtract(wristWorld));
        }
        this.referenceOffsets = offsets;
        this.referenceOrientation = orientation;
        this.lastHandednessScore = handLandmarks.handednessScore;
        this.lastPresenceScore = handLandmarks.presenceScore;
    }

    /**
     * Rigidly rotates the cached reference hand shape by however much the IMU orientation has
     * changed since it was captured, anchors it at the fused wrist position, and converts the
     * result back to camera space (the frame {@code HandLandmarks.absolutePoints} — and therefore
     * DisplayApp / geometricNormal — expects).
     */
    private HandLandmarks reprojectHand(Point3D fusedWristWorld, ImuPacket.Quaternion orientation) {
        if (referenceOffsets == null) {
            return null;
        }

        double[][] delta = multiply3x3(worldRotationMatrix(orientation), transpose3x3(worldRotationMatrix(referenceOrientation)));

        List<Point3D> absolutePoints = new ArrayList<>(referenceOffsets.size());
        for (Point3D offset : referenceOffsets) {
            Point3D worldPoint = fusedWristWorld.add(applyMatrix(delta, offset));
            absolutePoints.add(flipYZ(worldPoint));
        }

        return new HandLandmarks(absolutePoints, lastHandednessScore, lastPresenceScore);
    }

    /**
     * Converts between {@code HandLandmarks.absolutePoints}' OpenCV camera-space convention
     * (X = right, Y = down, Z = forward/away from camera) and the IMU/world convention used for
     * accel fusion, {@link #worldRotationMatrix} and {@code AnalysisThread.M} (X = right, Y = up,
     * Z = toward camera). A pure axis flip, and its own inverse, so the same formula converts
     * either direction.
     */
    private static Point3D flipYZ(Point3D p) {
        return new Point3D(p.getX(), -p.getY(), -p.getZ());
    }

    /**
     * Raw accelerometer packet converted to sensor-frame specific force (m/s^2). Rotation into
     * world frame, gravity compensation, and accelerometer-bias cancellation all now happen
     * inside {@link UKF.WristFilter}'s process model instead of here, since the bias correction
     * must be applied per-sigma-point (using that sigma point's own bias hypothesis) for the
     * filter to be able to observe and estimate the bias at all.
     *
     * <p>Previously negated the raw vector to work around gravity doubling instead of
     * cancelling. That was papering over {@code M} having the wrong handedness (a reflection,
     * det -1, instead of a proper rotation) — fixed directly in {@code M} now, so this no longer
     * needs a compensating sign flip. Re-verify gravity cancellation on-device after this change.
     */
    private static Point3D sensorAccelerationMps2(ImuPacket.Acceleration accel) {
        return new Point3D(accel.ax(), accel.ay(), accel.az()).multiply(GRAVITY_MPS2);
    }

    // -------------------------------------------------------------------------
    // IMU arrow
    // -------------------------------------------------------------------------

    /**
     * Fixed mounting correction matrix M.
     *
     * <p>Encodes the physical orientation of the sensor relative to the world frame, as measured
     * in the sensor's reference pose (roll = pitch = yaw = 0). Each column is where the
     * corresponding sensor basis vector points in world coordinates at that pose:
     * <pre>
     *   Sensor +X → world -Z  (away from camera)
     *   Sensor +Y → world -X  (camera's left)
     *   Sensor +Z → world +Y  (up)
     * </pre>
     *
     * <pre>
     *       sX  sY  sZ
     *   M = [ 0  -1   0 ]   world X
     *       [ 0   0   1 ]   world Y
     *       [-1   0   0 ]   world Z
     * </pre>
     */
    public static final double[][] M = {
            { 0, -1,  0 },
            { 0,  0,  1 },
            {-1,  0,  0 },
    };

    /**
     * Computes the IMU sensor's normal vector in world coordinates.
     *
     * <p>Applies {@code n_world = M * R(quaternion) * (0,0,-1)}, where:
     * <ul>
     *   <li>{@code (0,0,-1)} is the sensor's outward normal in its own frame (-Z face)</li>
     *   <li>{@code R} is the rotation the WIT sensor's quaternion represents</li>
     *   <li>{@code M} is the fixed mounting correction matrix above</li>
     * </ul>
     *
     * @param orientation IMU quaternion packet
     * @return unit normal vector in world coordinates
     */
    public static Point3D imuNormal(ImuPacket.Quaternion orientation) {
        return rotateSensorToWorld(new Point3D(0, 0, -1), orientation);
    }

    /**
     * Full sensor-to-world rotation matrix {@code M * R(quaternion)}, generalizing the
     * {@code n_world = M * R * (0,0,-1)} formula above to arbitrary sensor-frame vectors
     * (used both for {@link #imuNormal} and to rotate IMU acceleration / the rest of the
     * hand's landmarks into world coordinates in {@code kalmanFuse}).
     *
     * <p>Builds {@code R} directly from the quaternion (standard conversion formula, no
     * trigonometry) instead of round-tripping through Euler roll/pitch/yaw. That round trip
     * has a gimbal-lock singularity at pitch = ±90° — a pose a wrist-mounted sensor reaches
     * easily (forearm held vertical) — where roll and yaw become numerically indistinguishable
     * and the reconstructed matrix degrades. The quaternion has no such singularity, and skips
     * the conversion's two lossy trigonometric round trips per call.
     */
    private static double[][] worldRotationMatrix(ImuPacket.Quaternion orientation) {
        double qw = orientation.qw();
        double qx = orientation.qx();
        double qy = orientation.qy();
        double qz = orientation.qz();

        double[][] rot = {
                { 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw) },
                { 2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw) },
                { 2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy) }
        };

        return multiply3x3(M, rot);
    }

    private static Point3D rotateSensorToWorld(Point3D sensorVec, ImuPacket.Quaternion orientation) {
        return applyMatrix(worldRotationMatrix(orientation), sensorVec);
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
        int totalPoints = landmarks.absolutePoints.size();

        for (Point3D p : landmarks.absolutePoints) {
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


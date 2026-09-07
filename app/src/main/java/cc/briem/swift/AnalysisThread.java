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

    /**
     * Camera mount tilt about the camera's own X (right) axis, in radians: positive = camera
     * pitched down toward the table (forward axis dips below horizontal), negative = pitched up
     * toward the ceiling, 0 = level. Everything else in this class ({@link #M}, gravity
     * compensation in {@code UKF}, and {@link #yawOffsetMatrix}'s heading-only correction) assumes
     * the camera's forward axis is horizontal — see {@link #cameraToWorld} for how this constant
     * folds the actual mount angle back into that assumption instead of leaving it violated.
     * Yaw (facing a different horizontal direction) is already corrected at runtime instead of
     * needing a constant here, and roll (rotation about the optical axis) never happens on this
     * rig, so tilt is the only mounting angle that needs to be dialed in by hand.
     */
    private volatile double cameraTiltRadians = 0.0;

    /**
     * Sets the camera's mount tilt — see {@link #cameraTiltRadians}. Call once after construction
     * to match however the camera is physically angled this session (e.g. 45 when angled down at
     * a checkerboard for occlusion testing); leave at the default 0 for a level, forward-facing
     * mount.
     */
    public void setCameraTiltDegrees(double degrees) {
        this.cameraTiltRadians = Math.toRadians(degrees);
    }

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
                    if (isConfidentDetection(rawHand)) {
                        // Refine the IMU->world yaw offset against this tick's CV ground truth —
                        // see updateYawOffset.
                        updateYawOffset(geometricNormal[0], rawImuNormal(quaternionPacket));
                    }
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

        boolean confident = isConfidentDetection(handLandmarks);
        List<Point3D> absolutePoints = confident ? handLandmarks.absolutePoints : null;
        Point3D wristWorld = confident ? cameraToWorld(absolutePoints.get(0)) : null;

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
     * the IMU/world convention via {@link #cameraToWorld} so the rigid rotation in
     * {@link #reprojectHand} lines up with {@link #worldRotationMatrix}.
     */
    private void captureReferenceShape(List<Point3D> absolutePoints, ImuPacket.Quaternion orientation, HandLandmarks handLandmarks) {
        Point3D wristWorld = cameraToWorld(absolutePoints.get(0));
        List<Point3D> offsets = new ArrayList<>(absolutePoints.size());
        for (Point3D p : absolutePoints) {
            offsets.add(cameraToWorld(p).subtract(wristWorld));
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
            absolutePoints.add(worldToCamera(worldPoint));
        }

        return new HandLandmarks(absolutePoints, lastHandednessScore, lastPresenceScore);
    }

    /** Shared confidence gate for a raw CV detection — "trust this frame as absolute truth". */
    private static boolean isConfidentDetection(HandLandmarks h) {
        return h != null && h.absolutePoints != null && !h.absolutePoints.isEmpty()
                && h.presenceScore >= PRESENCE_THRESHOLD;
    }

    /**
     * Converts a point from {@code HandLandmarks.absolutePoints}' OpenCV camera-space convention
     * (X = right, Y = down, Z = forward/away from camera) to the IMU/world convention used for
     * accel fusion, {@link #worldRotationMatrix} and {@code AnalysisThread.M} (X = right, Y = up,
     * Z = toward camera).
     *
     * <p>At zero tilt this is the pure axis flip {@code (x, -y, -z)} it always used to be — a 180°
     * rotation about the shared X axis. {@link #cameraTiltRadians} generalizes that fixed 180° to
     * {@code 180° - tilt}, which un-tilts the camera's actual forward axis back to horizontal
     * before applying the flip, so the result is a genuinely gravity/level-referenced world frame
     * regardless of how the camera is physically angled. Unlike the old pure flip, this is not its
     * own inverse once tilt != 0 — see {@link #worldToCamera} for the reverse direction.
     */
    private Point3D cameraToWorld(Point3D p) {
        double a = Math.PI - cameraTiltRadians;
        double c = Math.cos(a), s = Math.sin(a);
        double x = p.getX(), y = p.getY(), z = p.getZ();
        return new Point3D(x, c * y - s * z, s * y + c * z);
    }

    /**
     * Inverse of {@link #cameraToWorld}: converts a fused/world-frame point back to camera space
     * for rendering (e.g. {@link #reprojectHand}). {@link #cameraToWorld}'s matrix is a proper
     * rotation, so its inverse is just its transpose.
     */
    private Point3D worldToCamera(Point3D p) {
        double a = Math.PI - cameraTiltRadians;
        double c = Math.cos(a), s = Math.sin(a);
        double x = p.getX(), y = p.getY(), z = p.getZ();
        return new Point3D(x, c * y + s * z, -s * y + c * z);
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
     *
     * <p><b>Only roll/pitch (the vertical part) of this is trustworthy as a fixed constant.</b>
     * Roll/pitch are gravity-referenced, so they're the same physical relationship regardless of
     * where or when the sensor is used. Yaw is not: this WIT sensor's quaternion reports an
     * <i>absolute</i>, magnetometer-referenced heading (confirmed on-device — power-cycling in a
     * different orientation gives a different, not reset, yaw), so "yaw = 0" is a fixed direction
     * in the real world (e.g. magnetic north) that has no fixed relationship to "facing the
     * camera" — that relationship depends on which way the camera/desk happens to be pointing
     * that session. Baking a specific one in here would silently break in any other room/desk
     * orientation. So {@code M}'s yaw is left as whatever the reference-pose derivation above
     * produced, and corrected at runtime instead — see {@link #yawOffsetCos}.
     */
    public static final double[][] M = {
            { 0, -1,  0 },
            { 0,  0,  1 },
            {-1,  0,  0 },
    };

    // Runtime yaw calibration: corrects M's heading component by comparing the IMU-derived palm
    // normal (uncorrected) against the CV-derived one (ground truth, see geometricNormal)
    // whenever a confident CV frame is available — see updateYawOffset. Tracked as (cos, sin)
    // rather than a raw angle so blending across samples handles the ±180° wraparound correctly.
    private double yawOffsetCos = 1.0;
    private double yawOffsetSin = 0.0;
    private boolean yawOffsetCalibrated = false;

    private static final double YAW_CALIBRATION_ALPHA = 0.05; // low-pass factor, damps single-frame hand-pose noise
    private static final double MIN_HORIZONTAL_NORM = 0.15;   // below this a normal is too close to vertical to constrain yaw

    /**
     * Computes the IMU sensor's normal vector in world coordinates.
     *
     * <p>Applies {@code n_world = yawOffset * M * R(quaternion) * (0,0,-1)}, where:
     * <ul>
     *   <li>{@code (0,0,-1)} is the sensor's outward normal in its own frame (-Z face)</li>
     *   <li>{@code R} is the rotation the WIT sensor's quaternion represents</li>
     *   <li>{@code M} is the fixed mounting correction matrix above</li>
     *   <li>{@code yawOffset} is the runtime-calibrated heading correction, see {@link #yawOffsetCos}</li>
     * </ul>
     *
     * @param orientation IMU quaternion packet
     * @return unit normal vector in world coordinates
     */
    private Point3D imuNormal(ImuPacket.Quaternion orientation) {
        return rotateSensorToWorld(new Point3D(0, 0, -1), orientation);
    }

    /**
     * {@link #imuNormal}, but using {@code M} alone — without the runtime yaw correction. This is
     * the raw signal {@link #updateYawOffset} calibrates against; computing the calibrated
     * {@link #imuNormal} from CV ground truth would be circular.
     */
    private static Point3D rawImuNormal(ImuPacket.Quaternion orientation) {
        return applyMatrix(rotationBeforeYawCorrection(orientation), new Point3D(0, 0, -1));
    }

    /**
     * Refines {@link #yawOffsetCos}/{@link #yawOffsetSin} by comparing this tick's CV-truth palm
     * normal against {@link #rawImuNormal}. Only the horizontal (world X/Z) component of each
     * normal constrains heading, so this is skipped when either is too close to vertical (hand
     * facing straight up/down) for that angle to be numerically stable. Snaps directly to the
     * first usable sample (nothing to blend with yet), then low-pass filters further samples.
     */
    private void updateYawOffset(Point3D geometricNormal, Point3D rawImuNormal) {
        double geomHoriz = Math.hypot(geometricNormal.getX(), geometricNormal.getZ());
        double imuHoriz = Math.hypot(rawImuNormal.getX(), rawImuNormal.getZ());
        if (geomHoriz < MIN_HORIZONTAL_NORM || imuHoriz < MIN_HORIZONTAL_NORM) {
            return;
        }

        double geomHeading = Math.atan2(geometricNormal.getX(), geometricNormal.getZ());
        double imuHeading = Math.atan2(rawImuNormal.getX(), rawImuNormal.getZ());
        double sampleOffset = geomHeading - imuHeading;
        double sampleCos = Math.cos(sampleOffset);
        double sampleSin = Math.sin(sampleOffset);

        if (!yawOffsetCalibrated) {
            yawOffsetCos = sampleCos;
            yawOffsetSin = sampleSin;
            yawOffsetCalibrated = true;
        } else {
            yawOffsetCos += YAW_CALIBRATION_ALPHA * (sampleCos - yawOffsetCos);
            yawOffsetSin += YAW_CALIBRATION_ALPHA * (sampleSin - yawOffsetSin);
        }
    }

    /** Rotation about world Y (up) by the calibrated yaw offset — see {@link #yawOffsetCos}. */
    private double[][] yawOffsetMatrix() {
        return new double[][]{
                { yawOffsetCos, 0, yawOffsetSin },
                { 0,            1, 0            },
                {-yawOffsetSin, 0, yawOffsetCos },
        };
    }

    /**
     * Full sensor-to-world rotation matrix {@code yawOffset * M * R(quaternion)}, generalizing
     * the {@code n_world = M * R * (0,0,-1)} formula above to arbitrary sensor-frame vectors
     * (used both for {@link #imuNormal} and to rotate IMU acceleration / the rest of the
     * hand's landmarks into world coordinates in {@code kalmanFuse}).
     */
    private double[][] worldRotationMatrix(ImuPacket.Quaternion orientation) {
        return multiply3x3(yawOffsetMatrix(), rotationBeforeYawCorrection(orientation));
    }

    /**
     * {@code M * R(quaternion)}, before the runtime yaw correction — see {@link #rawImuNormal}
     * and {@link #worldRotationMatrix}.
     *
     * <p>Builds {@code R} directly from the quaternion (standard conversion formula, no
     * trigonometry) instead of round-tripping through Euler roll/pitch/yaw. That round trip
     * has a gimbal-lock singularity at pitch = ±90° — a pose a wrist-mounted sensor reaches
     * easily (forearm held vertical) — where roll and yaw become numerically indistinguishable
     * and the reconstructed matrix degrades. The quaternion has no such singularity, and skips
     * the conversion's two lossy trigonometric round trips per call.
     */
    private static double[][] rotationBeforeYawCorrection(ImuPacket.Quaternion orientation) {
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

    private Point3D rotateSensorToWorld(Point3D sensorVec, ImuPacket.Quaternion orientation) {
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


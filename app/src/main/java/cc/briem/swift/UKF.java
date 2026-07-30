package cc.briem.swift;

import java.util.ArrayList;
import java.util.List;

import cc.briem.swift.cv.models.HandLandmarks;
import javafx.geometry.Point3D;

public class UKF {

    /**
     * Calibrated camera intrinsic parameters.
     *
     * <p>Obtain these values by running a standard camera calibration procedure
     * (e.g. OpenCV {@code calibrateCamera} with a checkerboard pattern).
     *
     * <pre>
     *   fx, fy  — focal lengths in pixels
     *   cx, cy  — principal point (optical centre) in pixels
     * </pre>
     */
    public record CameraIntrinsics(double fx, double fy, double cx, double cy) {

        public static final CameraIntrinsics MBP = new CameraIntrinsics(
                1500.0 , // fx (e.g. 921.5)
                1500.0, // fy (e.g. 921.5)
                960.0 , // cx (e.g. 640.0 for a 1280-wide image)
                540.0  // cy (e.g. 360.0 for a  720-tall image)
        );
    }

    private static final int PALM_REF_A = 0;
    private static final int PALM_REF_B = 9;

    /**
     * Rough average adult palm width (landmark 0 → 9) in metres, used as the default metric
     * scale reference for {@link #toWorldCoordinates}. Monocular depth is scale-ambiguous, so
     * this is only an approximation — calibrate per-user for better absolute accuracy.
     */
    public static final double DEFAULT_PALM_WIDTH_METRES = 0.08;

    public static HandLandmarks toWorldCoordinates(
            HandLandmarks landmarks,
            CameraIntrinsics intrinsics,
            double palmWidthMetres) {

        // --- Step 1: compute metric depth scale ---------------------------------
        Point3D refA = landmarks.points.get(PALM_REF_A);
        Point3D refB = landmarks.points.get(PALM_REF_B);

        // Projected palm width: 2-D Euclidean distance in the image plane (x,y only).
        double dx = refB.getX() - refA.getX();
        double dy = refB.getY() - refA.getY();
        double palmWidthPx = Math.sqrt(dx * dx + dy * dy);

        if (palmWidthPx == 0.0) {
            return null;
        }

        double depthScale = (palmWidthMetres * intrinsics.fx()) / palmWidthPx;

        List<Point3D> worldPoints = new ArrayList<>(landmarks.points.size());

        for (Point3D p : landmarks.points) {
            double zCam = p.getZ() * depthScale;
            double xCam = (p.getX() - intrinsics.cx()) / intrinsics.fx() * zCam;
            double yCam = (p.getY() - intrinsics.cy()) / intrinsics.fy() * zCam;

            worldPoints.add(new Point3D(xCam, -yCam, -zCam));
        }

        return new HandLandmarks(worldPoints, landmarks.worldPoints, landmarks.handednessScore, landmarks.presenceScore);
    }

    /**
     * Sigma-point (unscented) Kalman filter tracking the wrist's 3-D position, velocity, and
     * accelerometer bias.
     *
     * <p>State vector: {@code [px, py, pz, vx, vy, vz, bx, by, bz]}, in the same metric world
     * frame as {@link #toWorldCoordinates} and {@code AnalysisThread.M} (X = camera right,
     * Y = up, Z = toward camera) for position/velocity; {@code bx,by,bz} is the accelerometer's
     * bias in its own sensor frame (m/s^2).
     *
     * <p>The bias exists because IMU-only dead reckoning (no CV correction available, e.g. during
     * occlusion) double-integrates whatever the accelerometer reports, and any small constant
     * sensor bias grows unboundedly under double integration — this showed up as the tracked
     * position drifting away over an occlusion, badly enough that its estimated depth would blow
     * up and (via perspective projection) make the whole hand appear to collapse toward a point.
     * Modeling the bias as part of the state lets {@link #update} correct it too (via its
     * cross-covariance with position, built up over repeated predict/update cycles) whenever a
     * confident CV frame is available, so it's actually learned and cancelled rather than left to
     * accumulate — the standard technique for this failure mode in INS/GPS fusion.
     *
     * <p>{@link #predict} advances the state using the raw sensor-frame specific force and the
     * current sensor-to-world rotation; bias subtraction and rotation happen per-sigma-point
     * (not precomputed) specifically so the filter can observe the bias's effect on predicted
     * position. It also applies an exponential velocity decay (a "singer"/drag-style term) so
     * that even a not-yet-fully-learned bias can't drive runaway velocity growth over an extended
     * gap between corrections. {@link #update} corrects the whole state against an absolute CV
     * wrist measurement. Both steps use the standard Van der Merwe scaled sigma-point set
     * (2N+1 = 19 points for N=9).
     */
    public static final class WristFilter {

        private static final int N = 9;

        // Van der Merwe scaled sigma-point parameters (standard defaults).
        private static final double ALPHA = 1e-3;
        private static final double BETA = 2.0;
        private static final double KAPPA = 0.0;
        private static final double LAMBDA = ALPHA * ALPHA * (N + KAPPA) - N;

        private static final double GRAVITY_MPS2 = 9.80665;

        // Tunable noise/dynamics.
        // accelNoiseStd (m/s^2): how much we distrust the constant-acceleration assumption between
        //   IMU samples, beyond what the bias state already accounts for.
        // measurementNoiseStd (m): how much we trust a confident CV wrist detection ("absolute truth").
        // biasRandomWalkStd (m/s^2 per sqrt(second)): how fast the accelerometer bias is allowed to
        //   drift — small, since real sensor bias changes slowly (temperature etc.), but non-zero so
        //   the filter can keep refining/adapting its estimate over time.
        // velocityDampingTauSeconds: exponential decay time constant applied to velocity every
        //   predict() call, absent that being corrected by measurement updates. Bounds how far
        //   uncorrected velocity (and therefore position) can run away during a long occlusion,
        //   without zeroing out real short-term motion between frequent confident corrections.
        private double accelNoiseStd = 1.5;
        private double measurementNoiseStd = 0.005;
        private double biasRandomWalkStd = 0.02;
        private double velocityDampingTauSeconds = 1.5;

        private double[] x = new double[N];
        private double[][] P;

        public WristFilter(Point3D initialPosition) {
            x[0] = initialPosition.getX();
            x[1] = initialPosition.getY();
            x[2] = initialPosition.getZ();
            P = identity(N, 0.01);            // ~10cm initial position uncertainty
            for (int i = 3; i < 6; i++) P[i][i] = 1.0;  // ~1 m/s initial velocity uncertainty
            for (int i = 6; i < 9; i++) P[i][i] = 4.0;  // ~2 m/s^2 initial bias uncertainty (unknown at start)
        }

        public void setAccelNoiseStd(double std) { this.accelNoiseStd = std; }

        public void setMeasurementNoiseStd(double std) { this.measurementNoiseStd = std; }

        public void setBiasRandomWalkStd(double std) { this.biasRandomWalkStd = std; }

        public void setVelocityDampingTauSeconds(double tauSeconds) { this.velocityDampingTauSeconds = tauSeconds; }

        public Point3D getPosition() { return new Point3D(x[0], x[1], x[2]); }

        public Point3D getVelocity() { return new Point3D(x[3], x[4], x[5]); }

        public Point3D getAccelBias() { return new Point3D(x[6], x[7], x[8]); }

        /**
         * Predict step: propagates the state forward by {@code dt} seconds given the raw
         * sensor-frame specific force {@code sensorAccelMps2} (m/s^2 — NOT yet bias-corrected,
         * rotated, or gravity-compensated: that all happens per-sigma-point below) and the
         * current sensor-to-world rotation matrix {@code worldRotation} (e.g. from
         * {@code AnalysisThread.worldRotationMatrix}).
         */
        public void predict(double dt, Point3D sensorAccelMps2, double[][] worldRotation) {
            if (dt <= 0) return;

            double[][] sigmas = sigmaPoints(x, P);
            double[][] propagated = new double[sigmas.length][];
            for (int i = 0; i < sigmas.length; i++) {
                propagated[i] = processModel(sigmas[i], dt, sensorAccelMps2, worldRotation);
            }

            double[] xPred = weightedMean(propagated);
            double[][] pPred = weightedCovariance(propagated, xPred, propagated, xPred, processNoise(dt));

            this.x = xPred;
            this.P = pPred;
        }

        /** Update step: corrects the state against an absolute wrist position measurement. */
        public void update(Point3D measuredPosition) {
            double[][] sigmas = sigmaPoints(x, P);
            double[][] Z = new double[sigmas.length][3];
            for (int i = 0; i < sigmas.length; i++) {
                Z[i][0] = sigmas[i][0];
                Z[i][1] = sigmas[i][1];
                Z[i][2] = sigmas[i][2];
            }

            double[] zPred = weightedMean(Z);
            double r2 = measurementNoiseStd * measurementNoiseStd;
            double[][] R = diag(new double[]{ r2, r2, r2 });
            double[][] Pzz = weightedCovariance(Z, zPred, Z, zPred, R);
            double[][] Pxz = weightedCovariance(sigmas, x, Z, zPred, null);

            double[][] K = multiply(Pxz, invert3x3(Pzz));

            double[] innovation = subtract(
                    new double[]{ measuredPosition.getX(), measuredPosition.getY(), measuredPosition.getZ() },
                    zPred);
            double[] correction = multiply(K, innovation);

            double[] xNew = new double[N];
            for (int i = 0; i < N; i++) xNew[i] = x[i] + correction[i];

            double[][] pNew = subtract(P, multiply(multiply(K, Pzz), transpose(K)));

            this.x = xNew;
            this.P = pNew;
        }

        // -- process model ------------------------------------------------------

        /**
         * Subtracts this sigma point's own bias hypothesis from the raw sensor reading, rotates
         * the result into world frame, adds gravity, then integrates position/velocity (with
         * velocity decay) and carries the bias through unchanged (it's a slow random walk,
         * modeled only via {@link #processNoise}, not driven deterministically here).
         */
        private double[] processModel(double[] s, double dt, Point3D sensorAccelMps2, double[][] worldRotation) {
            double correctedX = sensorAccelMps2.getX() - s[6];
            double correctedY = sensorAccelMps2.getY() - s[7];
            double correctedZ = sensorAccelMps2.getZ() - s[8];

            double worldAx = worldRotation[0][0] * correctedX + worldRotation[0][1] * correctedY + worldRotation[0][2] * correctedZ;
            double worldAy = worldRotation[1][0] * correctedX + worldRotation[1][1] * correctedY + worldRotation[1][2] * correctedZ;
            double worldAz = worldRotation[2][0] * correctedX + worldRotation[2][1] * correctedY + worldRotation[2][2] * correctedZ;
            worldAy -= GRAVITY_MPS2; // gravity points down (world -Y)

            double velocityDecay = Math.exp(-dt / velocityDampingTauSeconds);

            double[] out = new double[N];
            out[0] = s[0] + s[3] * dt;
            out[1] = s[1] + s[4] * dt;
            out[2] = s[2] + s[5] * dt;
            out[3] = (s[3] + worldAx * dt) * velocityDecay;
            out[4] = (s[4] + worldAy * dt) * velocityDecay;
            out[5] = (s[5] + worldAz * dt) * velocityDecay;
            out[6] = s[6];
            out[7] = s[7];
            out[8] = s[8];
            return out;
        }

        /** Discrete white-noise-acceleration process noise (pos/vel coupled per axis) plus a slow bias random walk. */
        private double[][] processNoise(double dt) {
            double q = accelNoiseStd * accelNoiseStd;
            double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
            double[][] Qm = new double[N][N];
            for (int axis = 0; axis < 3; axis++) {
                int p = axis, v = axis + 3;
                Qm[p][p] = q * dt4 / 4.0;
                Qm[p][v] = q * dt3 / 2.0;
                Qm[v][p] = q * dt3 / 2.0;
                Qm[v][v] = q * dt2;
            }
            double biasQ = biasRandomWalkStd * biasRandomWalkStd * dt;
            for (int axis = 0; axis < 3; axis++) {
                int b = axis + 6;
                Qm[b][b] = biasQ;
            }
            return Qm;
        }

        // -- sigma points / unscented transform ----------------------------------

        private static double[][] sigmaPoints(double[] mean, double[][] cov) {
            double[][] chol = cholesky(scale(cov, N + LAMBDA));
            double[][] sigmas = new double[2 * N + 1][N];
            sigmas[0] = mean.clone();
            for (int i = 0; i < N; i++) {
                double[] col = column(chol, i);
                for (int j = 0; j < N; j++) {
                    sigmas[i + 1][j] = mean[j] + col[j];
                    sigmas[N + i + 1][j] = mean[j] - col[j];
                }
            }
            return sigmas;
        }

        private static double meanWeight(int i) {
            return i == 0 ? LAMBDA / (N + LAMBDA) : 1.0 / (2.0 * (N + LAMBDA));
        }

        private static double covWeight(int i) {
            return i == 0 ? LAMBDA / (N + LAMBDA) + (1 - ALPHA * ALPHA + BETA) : 1.0 / (2.0 * (N + LAMBDA));
        }

        private static double[] weightedMean(double[][] sigmas) {
            int dim = sigmas[0].length;
            double[] mean = new double[dim];
            for (int i = 0; i < sigmas.length; i++) {
                double w = meanWeight(i);
                for (int j = 0; j < dim; j++) mean[j] += w * sigmas[i][j];
            }
            return mean;
        }

        private static double[][] weightedCovariance(double[][] a, double[] aMean, double[][] b, double[] bMean, double[][] noise) {
            int dimA = aMean.length, dimB = bMean.length;
            double[][] cov = new double[dimA][dimB];
            for (int i = 0; i < a.length; i++) {
                double w = covWeight(i);
                double[] da = subtract(a[i], aMean);
                double[] db = subtract(b[i], bMean);
                for (int r = 0; r < dimA; r++) {
                    for (int c = 0; c < dimB; c++) {
                        cov[r][c] += w * da[r] * db[c];
                    }
                }
            }
            if (noise != null) {
                for (int r = 0; r < dimA; r++) {
                    for (int c = 0; c < dimB; c++) {
                        cov[r][c] += noise[r][c];
                    }
                }
            }
            return cov;
        }

        // -- small dense matrix helpers (sized for N=6 / 3) ----------------------

        private static double[][] identity(int n, double scaleValue) {
            double[][] m = new double[n][n];
            for (int i = 0; i < n; i++) m[i][i] = scaleValue;
            return m;
        }

        private static double[][] diag(double[] values) {
            double[][] m = new double[values.length][values.length];
            for (int i = 0; i < values.length; i++) m[i][i] = values[i];
            return m;
        }

        private static double[][] scale(double[][] m, double s) {
            double[][] out = new double[m.length][m[0].length];
            for (int i = 0; i < m.length; i++)
                for (int j = 0; j < m[0].length; j++)
                    out[i][j] = m[i][j] * s;
            return out;
        }

        private static double[] column(double[][] m, int c) {
            double[] col = new double[m.length];
            for (int i = 0; i < m.length; i++) col[i] = m[i][c];
            return col;
        }

        private static double[] subtract(double[] a, double[] b) {
            double[] out = new double[a.length];
            for (int i = 0; i < a.length; i++) out[i] = a[i] - b[i];
            return out;
        }

        private static double[][] subtract(double[][] a, double[][] b) {
            double[][] out = new double[a.length][a[0].length];
            for (int i = 0; i < a.length; i++)
                for (int j = 0; j < a[0].length; j++)
                    out[i][j] = a[i][j] - b[i][j];
            return out;
        }

        private static double[][] transpose(double[][] m) {
            double[][] out = new double[m[0].length][m.length];
            for (int i = 0; i < m.length; i++)
                for (int j = 0; j < m[0].length; j++)
                    out[j][i] = m[i][j];
            return out;
        }

        private static double[][] multiply(double[][] a, double[][] b) {
            int n = a.length, k = b.length, m = b[0].length;
            double[][] out = new double[n][m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) {
                    double sum = 0;
                    for (int t = 0; t < k; t++) sum += a[i][t] * b[t][j];
                    out[i][j] = sum;
                }
            return out;
        }

        private static double[] multiply(double[][] a, double[] v) {
            double[] out = new double[a.length];
            for (int i = 0; i < a.length; i++) {
                double sum = 0;
                for (int j = 0; j < v.length; j++) sum += a[i][j] * v[j];
                out[i] = sum;
            }
            return out;
        }

        /** Lower-triangular Cholesky decomposition of a symmetric positive-(semi)definite matrix. */
        private static double[][] cholesky(double[][] a) {
            int n = a.length;
            double[][] l = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) sum += l[i][k] * l[j][k];
                    if (i == j) {
                        l[i][j] = Math.sqrt(Math.max(a[i][i] - sum, 0.0));
                    } else {
                        l[i][j] = (l[j][j] > 1e-12) ? (a[i][j] - sum) / l[j][j] : 0.0;
                    }
                }
            }
            return l;
        }

        /** Closed-form inverse of a 3x3 matrix. */
        private static double[][] invert3x3(double[][] m) {
            double a = m[0][0], b = m[0][1], c = m[0][2];
            double d = m[1][0], e = m[1][1], f = m[1][2];
            double g = m[2][0], h = m[2][1], i = m[2][2];

            double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
            double det = a * A + b * B + c * C;
            if (Math.abs(det) < 1e-12) det = 1e-12;

            double D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g);
            double G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d;

            return new double[][]{
                    { A / det, D / det, G / det },
                    { B / det, E / det, H / det },
                    { C / det, F / det, I / det }
            };
        }
    }
}

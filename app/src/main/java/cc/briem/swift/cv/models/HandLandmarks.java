package cc.briem.swift.cv.models;

import java.util.ArrayList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;

import javafx.geometry.Point3D;

public class HandLandmarks {
    public final List<Point3D> points;          // 21 (x,y,z) in image coordinates
    public final List<Point3D> worldPoints;     // 21 (x,y,z) in metric world coordinates (MediaPipe Identity_3)
    public final double handednessScore; // >0.5 -> right else left
    public final double presenceScore;

    public HandLandmarks(List<Point3D> points, List<Point3D> worldPoints, double handednessScore, double presenceScore) {
        this.points = points;
        this.worldPoints = worldPoints;
        this.handednessScore = handednessScore;
        this.presenceScore = presenceScore;
    }

    public HandLandmarks add(HandLandmarks other) {
        List<Point3D> newPoints = new ArrayList<>(this.points.size());
        for(int i = 0; i < this.points.size(); i++) {
            newPoints.add(i, this.points.get(i).add(other.points.get(i)));
        }

        return new HandLandmarks(newPoints, worldPoints, handednessScore, presenceScore);
    }

    public HandLandmarks subtract(HandLandmarks other) {
        List<Point3D> newPoints = new ArrayList<>(this.points.size());
        for(int i = 0; i < this.points.size(); i++) {
            newPoints.add(i, this.points.get(i).subtract(other.points.get(i)));
        }

        return new HandLandmarks(newPoints, worldPoints, handednessScore, presenceScore);
    }

    public HandLandmarks multiply(double other) {
        List<Point3D> newPoints = new ArrayList<>(this.points.size());
        for(int i = 0; i < this.points.size(); i++) {
            newPoints.add(i, this.points.get(i).multiply(other));
        }
        return new HandLandmarks(newPoints, worldPoints, handednessScore, presenceScore);
    }

    public HandLandmarks divide(double other) {
        return this.multiply(1.0 / other);
    }

    public List<Point3D> getAbsoluteWorldPoints() {

        // Camera Intrinsics
        final double width = 1920;
        final double height = 1080;
        final double fovHorizontal = 65;
        final double fx = (width / 2) / Math.tan(Math.toRadians(fovHorizontal) / 2);  // ≈ 1507
        final double fy = fx;
        final double cx = width / 2;
        final double cy = height / 2;

        Mat cameraMatrix = Mat.zeros(3, 3, CvType.CV_64FC1);
        cameraMatrix.put(0, 0, fx);
        cameraMatrix.put(0, 2, cx);
        cameraMatrix.put(1, 1, fy);
        cameraMatrix.put(1, 2, cy);
        cameraMatrix.put(2, 2, 1.0);

        MatOfDouble distCoeffs = new MatOfDouble(0, 0, 0, 0);
        MatOfPoint3f objectPoints = new MatOfPoint3f();
        MatOfPoint2f imagePoints  = new MatOfPoint2f();
        Mat rvec = new Mat();
        Mat tvec = new Mat();
        Mat rotation = new Mat();

        try {
            // Convert Point3D to Point3 (OpenCV)
            List<Point3> objList = new ArrayList<>(this.worldPoints.size());
            for (Point3D wp : this.worldPoints) {
                objList.add(new Point3(wp.getX(), wp.getY(), wp.getZ()));
            }
            objectPoints.fromList(objList);
            List<Point> imgList = new ArrayList<>(this.points.size());
            for (Point3D p : this.points) {
                imgList.add(new Point(p.getX(), p.getY()));
            }
            imagePoints.fromList(imgList);

            // Get Z from solvePnP
            boolean solved = Calib3d.solvePnP(
                objectPoints, imagePoints, cameraMatrix, distCoeffs,
                rvec, tvec, false, Calib3d.SOLVEPNP_SQPNP
            );

            if (!solved) {
                return List.of();
            }

            Calib3d.Rodrigues(rvec, rotation);

            double[][] r = new double[3][3];
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    r[row][col] = rotation.get(row, col)[0];
                }
            }
            double tx = tvec.get(0, 0)[0];
            double ty = tvec.get(1, 0)[0];
            double tz = tvec.get(2, 0)[0];

            List<Point3D> absolute = new ArrayList<>(this.worldPoints.size());
            for (Point3D wp : this.worldPoints) {
                double x = wp.getX(),  y = wp.getY(), z = wp.getZ();
                absolute.add(new Point3D(
                    r[0][0] * x + r[0][1] * y + r[0][2] * z + tx,
                    r[1][0] * x + r[1][1] * y + r[1][2] * z + ty,
                    r[2][0] * x + r[2][1] * y + r[2][2] * z + tz));
            }
            return absolute;
        } finally {
            cameraMatrix.release();
            distCoeffs.release();
            objectPoints.release();
            imagePoints.release();
            rvec.release();
            tvec.release();
            rotation.release();
        }
    }
}
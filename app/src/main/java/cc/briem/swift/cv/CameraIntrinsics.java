package cc.briem.swift.cv;

import javafx.geometry.Point2D;
import javafx.geometry.Point3D;

public class CameraIntrinsics {
    public static final double width = 3840;
    public static final double height = 2160;
    public static final double fovHorizontal = 89.7;
    public static final double fx = (width / 2) / Math.tan(Math.toRadians(fovHorizontal) / 2);  // ≈ 966
    public static final double fy = fx;
    public static final double cx = width / 2;
    public static final double cy = height / 2;

    public static Point2D project(Point3D point) {
        if(point.getZ() <= 1e-6) {
            return null;
        }
        return new Point2D(
                fx * point.getX() / point.getZ() + cx,
                fy * point.getY() / point.getZ() + cy);
    }
}

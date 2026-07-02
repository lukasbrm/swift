package cc.briem.swift.cv;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;

public class AffineUtil {

    public static class Point2D {
        public double x, y;
        public Point2D(double x, double y) { this.x = x; this.y = y; }
    }

    /** 2x3 affine matrix: [m00 m01 m02; m10 m11 m12] - same layout as OpenCV's. */
    public static class Mat2x3 {
        public double m00, m01, m02, m10, m11, m12;
        public Mat2x3(double m00, double m01, double m02, double m10, double m11, double m12) {
            this.m00 = m00; this.m01 = m01; this.m02 = m02;
            this.m10 = m10; this.m11 = m11; this.m12 = m12;
        }
    }

    /** Equivalent of cv2.getRotationMatrix2D(center, angleDeg, scale=1.0). */
    public static Mat2x3 rotationMatrix2D(double cx, double cy, double angleDeg) {
        double rad = Math.toRadians(angleDeg);
        double alpha = Math.cos(rad);
        double beta = Math.sin(rad);
        return new Mat2x3(
            alpha, beta, (1 - alpha) * cx - beta * cy,
            -beta, alpha, beta * cx + (1 - alpha) * cy
        );
    }

    /** Standard forward affine point transform: [x,y] -> M * [x,y,1]. */
    public static Point2D applyForward(Mat2x3 m, double x, double y) {
        return new Point2D(m.m00 * x + m.m01 * y + m.m02, m.m10 * x + m.m11 * y + m.m12);
    }

    /** General 2x3 affine inverse. */
    public static Mat2x3 invert(Mat2x3 m) {
        double det = m.m00 * m.m11 - m.m01 * m.m10;
        double i00 = m.m11 / det, i01 = -m.m01 / det;
        double i10 = -m.m10 / det, i11 = m.m00 / det;
        double i02 = -(i00 * m.m02 + i01 * m.m12);
        double i12 = -(i10 * m.m02 + i11 * m.m12);
        return new Mat2x3(i00, i01, i02, i10, i11, i12);
    }

    /**
     * Equivalent of cv2.warpAffine(image, M, sameSize) with black (constant) border,
     * i.e. forward-maps src pixels into dst using M, canvas stays the same size as src.
     */
    public static BufferedImage warpAffine(BufferedImage src, Mat2x3 m) {
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = dst.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, dst.getWidth(), dst.getHeight());
        AffineTransform t = new AffineTransform(m.m00, m.m10, m.m01, m.m11, m.m02, m.m12);
        g.drawImage(src, t, null);
        g.dispose();
        return dst;
    }
}
package cc.briem.swift;

import java.util.List;

import cc.briem.swift.cv.models.HandLandmarks;
import cc.briem.swift.cv.models.LandmarkResult;

import cc.briem.swift.imu.ImuPacket;
import javafx.application.Application;
import javafx.geometry.Point3D;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class DisplayApp extends Application {

    static volatile ImageView imageView;
    static volatile Canvas overlay;

    private static final double DOT_RADIUS        = 5.0;
    private static final double LINE_WIDTH        = 2.0;
    private static final Color  DOT_COLOR         = Color.LIME;
    private static final Color  LINE_COLOR        = Color.WHITE;
    private static final double ARROW_LENGTH      = 120.0;
    private static final double ARROW_HEAD_LENGTH = 18.0;
    private static final double ARROW_HEAD_ANGLE  = Math.toRadians(30);
    private static final Color  ARROW_COLOR       = Color.CYAN;

    // MediaPipe hand skeleton: chains of landmark indices per finger, plus the palm bar.
    private static final int[][] CONNECTIONS = {
        {0, 1, 2, 3, 4},
        {0, 5, 6, 7, 8},
        {0, 9, 10, 11, 12},
        {0, 13, 14, 15, 16},
        {0, 17, 18, 19, 20},
        {5, 9, 13, 17},
    };

    @Override
    public void start(Stage primaryStage) {
        imageView = new ImageView();
        imageView.setPreserveRatio(true);
        imageView.setSmooth(true);

        overlay = new Canvas(1280, 720);

        // overlay is added second so it renders on top of imageView
        StackPane root = new StackPane(imageView, overlay);
        Scene scene = new Scene(root, 1280, 720);

        imageView.fitWidthProperty().bind(scene.widthProperty());
        imageView.fitHeightProperty().bind(scene.heightProperty());
        overlay.widthProperty().bind(scene.widthProperty());
        overlay.heightProperty().bind(scene.heightProperty());

        primaryStage.setTitle("Swift Frame Viewer");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * Updates the camera frame and redraws the landmark overlay.
     * Must be called on the JavaFX Application Thread.
     */
    public static void render(Image image, LandmarkResult result, ImuPacket.Angle anglePacket) {
        if (imageView != null) imageView.setImage(image);
        drawOverlay(image, result, anglePacket);
    }

    private static void drawOverlay(Image image, LandmarkResult result, ImuPacket.Angle anglePacket) {
        if (overlay == null) return;

        GraphicsContext gc = overlay.getGraphicsContext2D();
        double canvasW = overlay.getWidth();
        double canvasH = overlay.getHeight();

        gc.clearRect(0, 0, canvasW, canvasH);

        if (result == null || result.getHands().isEmpty()) return;

        // The ImageView letterboxes the image to preserve aspect ratio.
        // Compute the scale and offset so landmarks map to the correct canvas position.
        double scale = Math.min(canvasW / image.getWidth(), canvasH / image.getHeight());
        double offsetX = (canvasW - image.getWidth()  * scale) / 2.0;
        double offsetY = (canvasH - image.getHeight() * scale) / 2.0;

        for (HandLandmarks hand : result.getHands()) {
            List<Point3D> pts = hand.points;

            // Draw skeleton lines beneath the dots
            gc.setStroke(LINE_COLOR);
            gc.setLineWidth(LINE_WIDTH);
            for (int[] chain : CONNECTIONS) {
                for (int i = 0; i < chain.length - 1; i++) {
                    Point3D a = pts.get(chain[i]);
                    Point3D b = pts.get(chain[i + 1]);
                    gc.strokeLine(
                        a.getX() * scale + offsetX, a.getY() * scale + offsetY,
                        b.getX() * scale + offsetX, b.getY() * scale + offsetY
                    );
                }
            }

            // Draw a dot at each of the 21 landmark positions
            gc.setFill(DOT_COLOR);
            for (Point3D p : pts) {
                double cx = p.getX() * scale + offsetX;
                double cy = p.getY() * scale + offsetY;
                gc.fillOval(cx - DOT_RADIUS, cy - DOT_RADIUS, DOT_RADIUS * 2, DOT_RADIUS * 2);
            }

            // Draw IMU orientation arrow rooted at the wrist
            if (anglePacket != null) {
                drawImuArrow(gc, pts, anglePacket, scale, offsetX, offsetY);
            }
        }
    }

    /**
     * Computes the IMU sensor's normal vector in world coordinates.
     *
     * <p>The sensor's normal face is {@code -Z}, i.e. {@code (0, 0, -1)} at rest (flat, face-up).
     * Rotation convention is ZYX Euler as used by WIT sensors. Applying
     * {@code R = Rz(yaw) * Ry(pitch) * Rx(roll)} to {@code (0, 0, -1)} yields the negated
     * third column of R. Yaw drops out analytically since it only rotates around Z.
     * <pre>
     *   nx = -sin(pitch)
     *   ny =  cos(pitch) * sin(roll)
     *   nz = -cos(pitch) * cos(roll)
     * </pre>
     * Note: WIT firmware reports roll as the Y-axis rotation and pitch as the X-axis rotation,
     * which is the opposite of the mathematical convention — they are swapped here accordingly.
     *
     * @param angle IMU angle packet (roll, pitch, yaw in degrees)
     * @return unit normal vector in world coordinates
     */
    private static Point3D imuNormal(ImuPacket.Angle angle) {
        // WIT firmware swaps the conventional roll/pitch labels:
        // what it calls "pitch" rotates around X, "roll" rotates around Y.
        double roll  = Math.toRadians(angle.pitch());
        double pitch = Math.toRadians(angle.roll());

        double nx =  -Math.sin(pitch);
        double ny =   Math.cos(pitch) * Math.sin(roll);
        double nz =  -Math.cos(pitch) * Math.cos(roll);

        return new Point3D(nx, ny, nz);
    }

    /**
     * Draws an arrow on {@code gc} representing the IMU normal vector, rooted at the wrist landmark.
     *
     * <p>Projection rules:
     * <ul>
     *   <li>The world-space normal's X component maps directly to canvas X.</li>
     *   <li>The world-space normal's Y component is negated for canvas Y (image Y increases downward).</li>
     *   <li>The arrow length is modulated by {@code 1 - |nz|}: the arrow is shortest when the normal
     *       points directly at or away from the camera, and longest when it lies in the image plane.</li>
     * </ul>
     *
     * @param gc          graphics context (must be on the JavaFX Application Thread)
     * @param pts         21 hand landmark points in image pixel coordinates
     * @param anglePacket IMU angle packet providing roll/pitch/yaw
     * @param scale       letterbox scale factor (canvas pixels per image pixel)
     * @param offsetX     letterbox horizontal offset in canvas pixels
     * @param offsetY     letterbox vertical offset in canvas pixels
     */
    private static void drawImuArrow(GraphicsContext gc, List<Point3D> pts,
                                     ImuPacket.Angle anglePacket,
                                     double scale, double offsetX, double offsetY) {
        Point3D normal = imuNormal(anglePacket);

        // Wrist (landmark 0) in canvas coordinates
        double wx = pts.get(0).getX() * scale + offsetX;
        double wy = pts.get(0).getY() * scale + offsetY;

        // Foreshorten: arrow is shortest when normal faces the camera (|nz| → 1)
        double projectedLength = ARROW_LENGTH * (1.0 - Math.abs(normal.getZ()));

        // Y is flipped: world +Y (up) → canvas -Y (up on screen)
        double tx = wx + normal.getX()  * projectedLength;
        double ty = wy + (-normal.getY()) * projectedLength;

        // Shaft
        gc.setStroke(ARROW_COLOR);
        gc.setLineWidth(LINE_WIDTH + 1.0);
        gc.strokeLine(wx, wy, tx, ty);

        // Arrowhead: two lines from the tip, angled back toward the root
        double shaftAngle = Math.atan2(ty - wy, tx - wx);
        double leftAngle  = shaftAngle + Math.PI - ARROW_HEAD_ANGLE;
        double rightAngle = shaftAngle + Math.PI + ARROW_HEAD_ANGLE;

        gc.setLineWidth(1.0);
        gc.strokeLine(tx, ty,
                tx + Math.cos(leftAngle)  * ARROW_HEAD_LENGTH,
                ty + Math.sin(leftAngle)  * ARROW_HEAD_LENGTH);
        gc.strokeLine(tx, ty,
                tx + Math.cos(rightAngle) * ARROW_HEAD_LENGTH,
                ty + Math.sin(rightAngle) * ARROW_HEAD_LENGTH);
    }
}

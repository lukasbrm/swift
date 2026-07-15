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
import javafx.scene.paint.Paint;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DisplayApp extends Application {

    private static final Logger logger = LoggerFactory.getLogger(DisplayApp.class);

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

    // World-axes gizmo
    private static final double GIZMO_ORIGIN_X   = 60.0;   // px from left
    private static final double GIZMO_ORIGIN_Y   = 60.0;   // px from bottom
    private static final double GIZMO_AXIS_LEN   = 40.0;
    private static final Color  GIZMO_X_COLOR    = Color.RED;
    private static final Color  GIZMO_Y_COLOR    = Color.LIMEGREEN;
    private static final Color  GIZMO_Z_COLOR    = Color.DODGERBLUE;

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
    public static void render(Image image, LandmarkResult result, Point3D[] imuNormal, Point3D[] geometricNormal, TrackingState state) {
        if (imageView != null) imageView.setImage(image);
        drawOverlay(image, result, imuNormal, geometricNormal, state);
    }

    private static void drawOverlay(Image image, LandmarkResult result, Point3D[] imuNormal, Point3D[] geometricNormal, TrackingState state) {
        if (overlay == null) return;

        GraphicsContext gc = overlay.getGraphicsContext2D();
        double canvasW = overlay.getWidth();
        double canvasH = overlay.getHeight();

        gc.clearRect(0, 0, canvasW, canvasH);

        switch(state) {
            case TrackingState.TRACKING -> drawText(gc, "TRACKING", Color.GREEN, 20, 40);
            case TrackingState.MISMATCH -> drawText(gc, "MISMATCH", Color.ORANGE, 20, 40);
            case TrackingState.OCCLUDED -> drawText(gc, "OCCLUDED", Color.DARKRED, 20, 40);
        }

        // Always draw the world-axes gizmo so the coordinate assumptions are visible
        drawGizmo(gc, canvasW, canvasH);

        if (result == null || result.getHands().isEmpty()) return;

        // The ImageView letterboxes the image to preserve aspect ratio.
        // Compute the scale and offset so landmarks map to the correct canvas position.
        double scale   = Math.min(canvasW / image.getWidth(), canvasH / image.getHeight());
        double offsetX = (canvasW - image.getWidth()  * scale) / 2.0;
        double offsetY = (canvasH - image.getHeight() * scale) / 2.0;

        int handIndex = 0;
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
            drawArrow(gc, pts, imuNormal[handIndex], scale, offsetX, offsetY, Color.CYAN);

            // Draw geometric palm normal from landmarks
            drawArrow(gc, pts, geometricNormal[handIndex], scale, offsetX, offsetY, Color.GREEN);

            handIndex++;
        }
    }

    // -------------------------------------------------------------------------
    // World-axes gizmo
    // -------------------------------------------------------------------------

    /**
     * Draws a fixed world-axes gizmo in the bottom-left corner of the canvas.
     *
     * <p>Coordinate conventions shown:
     * <ul>
     *   <li><b>+X</b> (red)   → right on screen</li>
     *   <li><b>+Y</b> (green) → up on screen (world +Y is up; canvas Y is flipped)</li>
     *   <li><b>+Z</b> (blue)  → toward the camera, shown as a dot-in-circle (bull's-eye)</li>
     * </ul>
     */
    private static void drawGizmo(GraphicsContext gc, double canvasW, double canvasH) {
        double ox  = GIZMO_ORIGIN_X;
        double oy  = canvasH - GIZMO_ORIGIN_Y;
        double len = GIZMO_AXIS_LEN;

        gc.save();
        gc.setLineWidth(2.0);
        gc.setFont(Font.font("SansSerif", FontWeight.BOLD, 12));

        // +X → right
        gc.setStroke(GIZMO_X_COLOR);
        gc.setFill(GIZMO_X_COLOR);
        gc.strokeLine(ox, oy, ox + len, oy);
        drawGizmoHead(gc, ox, oy, ox + len, oy);
        gc.fillText("+X", ox + len + 4, oy + 4);

        // +Y → up (canvas Y is inverted, so world +Y = canvas −Y direction)
        gc.setStroke(GIZMO_Y_COLOR);
        gc.setFill(GIZMO_Y_COLOR);
        gc.strokeLine(ox, oy, ox, oy - len);
        drawGizmoHead(gc, ox, oy, ox, oy - len);
        gc.fillText("+Y", ox + 4, oy - len - 4);

        // +Z → toward camera: bull's-eye symbol (circle + centre dot)
        gc.setStroke(GIZMO_Z_COLOR);
        gc.setFill(GIZMO_Z_COLOR);
        double r = 8.0;
        gc.strokeOval(ox - r, oy - r, r * 2, r * 2);
        gc.fillOval(ox - 2.5, oy - 2.5, 5.0, 5.0);
        gc.fillText("+Z", ox + r + 4, oy + 4);

        // "world" label below the origin
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("SansSerif", FontWeight.NORMAL, 10));
        gc.fillText("world", ox - 10, oy + 20);

        gc.restore();
    }

    /** Draws a small arrowhead at the tip {@code (x2, y2)} of a gizmo axis. */
    private static void drawGizmoHead(GraphicsContext gc, double x1, double y1, double x2, double y2) {
        double angle  = Math.atan2(y2 - y1, x2 - x1);
        double len    = 8.0;
        double spread = Math.toRadians(25);
        gc.strokeLine(x2, y2,
                x2 + Math.cos(angle + Math.PI - spread) * len,
                y2 + Math.sin(angle + Math.PI - spread) * len);
        gc.strokeLine(x2, y2,
                x2 + Math.cos(angle + Math.PI + spread) * len,
                y2 + Math.sin(angle + Math.PI + spread) * len);
    }

    private static void drawArrow(GraphicsContext gc, List<Point3D> pts,
                                     Point3D vector,
                                     double scale, double offsetX, double offsetY,
                                     Paint color) {
        // Wrist (landmark 0) in canvas coordinates
        double wx = pts.get(0).getX() * scale + offsetX;
        double wy = pts.get(0).getY() * scale + offsetY;

        // Y is flipped: world +Y (up) → canvas -Y (up on screen)
        double tx = wx + vector.getX()        * ARROW_LENGTH;
        double ty = wy + (-vector.getY())     * ARROW_LENGTH;

        // Shaft
        gc.setStroke(color);
        gc.setLineWidth(LINE_WIDTH + 1.0);
        gc.strokeLine(wx, wy, tx, ty);

        // Arrowhead (thinner than shaft)
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

    private static void drawText(GraphicsContext gc, String text, Color color, int x, int y) {

        gc.setFont(Font.font("Verdana", FontWeight.BOLD, 30));
        gc.setFill(color);
        gc.setStroke(color);
        gc.setLineWidth(1);

        gc.fillText(text, x, y);
    }
}

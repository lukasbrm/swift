package cc.briem.swift;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import javax.imageio.ImageIO;

import cc.briem.swift.cv.CameraIntrinsics;
import cc.briem.swift.cv.models.HandLandmarks;
import cc.briem.swift.cv.models.LandmarkResult;
import cc.briem.swift.imu.ImuPacket;
import javafx.application.Application;
import javafx.geometry.Point2D;
import javafx.geometry.Point3D;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
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

    // Debug-capture directory for 'S'-triggered screenshots + landmark dumps — see captureDebugFrame.
    private static final Path DEBUG_CAPTURE_DIR = Paths.get("debug-captures");
    private static final DateTimeFormatter CAPTURE_TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS");

    // Cache of the most recently rendered frame's inputs, so the 'S' key can dump "whatever's on
    // screen right now" without AnalysisThread having to push anything extra. Both render() (via
    // Platform.runLater) and the key handler run on the FX thread, so no synchronization is needed
    // beyond the volatile visibility already used for imageView/overlay above.
    private static volatile LandmarkResult lastResult;
    private static volatile Point3D[] lastImuNormal;
    private static volatile Point3D[] lastGeometricNormal;
    private static volatile TrackingState lastState;

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

        scene.setOnKeyPressed(event -> {
            if (event.getCode() == KeyCode.S) {
                captureDebugFrame();
            }
        });

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

        lastResult = result;
        lastImuNormal = imuNormal;
        lastGeometricNormal = geometricNormal;
        lastState = state;
    }

    private static void drawOverlay(Image image, LandmarkResult result, Point3D[] imuNormal, Point3D[] geometricNormal, TrackingState state) {
        if (overlay == null) return;

        GraphicsContext gc = overlay.getGraphicsContext2D();
        double canvasW = overlay.getWidth();
        double canvasH = overlay.getHeight();

        gc.clearRect(0, 0, canvasW, canvasH);

        /**switch(state) {
            case TrackingState.TRACKING -> drawText(gc, "TRACKING", Color.GREEN, 20, 40);
            case TrackingState.MISMATCH -> drawText(gc, "MISMATCH", Color.ORANGE, 20, 40);
            case TrackingState.OCCLUDED -> drawText(gc, "OCCLUDED", Color.DARKRED, 20, 40);
        }*/

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
            List<Point3D> pts = hand.absolutePoints;

            // Draw skeleton lines beneath the dots
            gc.setStroke(LINE_COLOR);
            gc.setLineWidth(LINE_WIDTH);
            for (int[] chain : CONNECTIONS) {
                for (int i = 0; i < chain.length - 1; i++) {
                    Point3D a3d = pts.get(chain[i]);
                    Point3D b3d = pts.get(chain[i + 1]);
                    Point2D a = CameraIntrinsics.project(a3d);
                    Point2D b = CameraIntrinsics.project(b3d);
                    if(a == null || b == null) continue;
                    gc.strokeLine(
                        a.getX() * scale + offsetX, a.getY() * scale + offsetY,
                        b.getX() * scale + offsetX, b.getY() * scale + offsetY
                    );
                }
            }

            // Draw a dot at each of the 21 landmark positions
            gc.setFill(DOT_COLOR);
            for (Point3D point : pts) {
                Point2D p = CameraIntrinsics.project(point);
                if (p == null) continue;
                double cx = p.getX() * scale + offsetX;
                double cy = p.getY() * scale + offsetY;
                gc.setFill(depthColor(point.getZ(), 0.05, 0.7));
                gc.fillOval(cx - DOT_RADIUS, cy - DOT_RADIUS, DOT_RADIUS * 2, DOT_RADIUS * 2);
            }

            // Draw IMU orientation arrow rooted at the wrist
            //drawArrow(gc, pts, imuNormal[handIndex], scale, offsetX, offsetY, Color.CYAN);

            // Draw geometric palm normal from landmarks
            //drawArrow(gc, pts, geometricNormal[handIndex], scale, offsetX, offsetY, Color.GREEN);

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

    private static Color depthColor(double z, double zMin, double zMax) {
        double t = Math.clamp((z - zMin) / (zMax - zMin), 0, 1);
        return Color.hsb(200 * t, 0.85, 1.0);
    }

    // -------------------------------------------------------------------------
    // Debug capture ('S' key): dumps the current annotated frame + its landmark data, for
    // comparing predicted hand position against an external ground truth (e.g. a checkerboard).
    // -------------------------------------------------------------------------

    /**
     * Snapshots the composited scene (camera frame + skeleton/arrows/gizmo overlay, exactly as
     * currently displayed) to a PNG, and writes a same-named JSON file with the landmark data
     * behind that frame, into {@link #DEBUG_CAPTURE_DIR}. Runs entirely on the FX thread (key
     * events always do), matching {@link #render}, so the cached {@code last*} fields are safe to
     * read here without extra synchronization.
     */
    private static void captureDebugFrame() {
        if (overlay == null) {
            return;
        }

        String timestamp = CAPTURE_TIMESTAMP_FORMAT.format(LocalDateTime.now());
        try {
            Files.createDirectories(DEBUG_CAPTURE_DIR);

            Node root = overlay.getParent();
            WritableImage snapshot = root.snapshot(new SnapshotParameters(), null);
            Path pngPath = DEBUG_CAPTURE_DIR.resolve(timestamp + ".png");
            writePng(snapshot, pngPath);

            Path jsonPath = DEBUG_CAPTURE_DIR.resolve(timestamp + ".json");
            writeLandmarksJson(jsonPath, timestamp);

            logger.info("Saved debug capture: {} / {}", pngPath, jsonPath);
        } catch (IOException e) {
            logger.error("Failed to save debug capture '{}'", timestamp, e);
        }
    }

    /**
     * Converts a JavaFX {@link WritableImage} to a PNG file by copying pixels through a
     * {@link BufferedImage} — no {@code javafx.swing} module (and its {@code SwingFXUtils}
     * shortcut) is on the module path, so this is done by hand instead of pulling that in for one
     * debug feature.
     */
    private static void writePng(WritableImage image, Path path) throws IOException {
        int width = (int) Math.round(image.getWidth());
        int height = (int) Math.round(image.getHeight());
        BufferedImage buffered = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        PixelReader reader = image.getPixelReader();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                buffered.setRGB(x, y, reader.getArgb(x, y));
            }
        }
        ImageIO.write(buffered, "png", path.toFile());
    }

    /**
     * Writes the landmark data behind the current frame (per hand: the 21 fused
     * {@code absolutePoints} in camera-space meters, handedness/presence, and normals) alongside
     * the tracking state and current camera-tilt setting, so a capture is self-describing when
     * revisited later. Hand-rolled (no JSON library in this project) since the schema is small and
     * fixed.
     */
    private static void writeLandmarksJson(Path path, String captureTimestamp) throws IOException {
        LandmarkResult result = lastResult;
        Point3D[] imuNormal = lastImuNormal;
        Point3D[] geometricNormal = lastGeometricNormal;
        TrackingState state = lastState;

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"captureTimestamp\": \"").append(captureTimestamp).append("\",\n");
        sb.append("  \"frameTimestamp\": \"").append(result != null ? result.getTimestamp() : "null").append("\",\n");
        sb.append("  \"trackingState\": \"").append(state != null ? state.name() : "UNKNOWN").append("\",\n");
        sb.append("  \"cameraTiltDegrees\": ").append(App.CAMERA_TILT_DEGREES).append(",\n");
        sb.append("  \"hands\": [\n");

        List<HandLandmarks> hands = result != null ? result.getHands() : List.of();
        for (int h = 0; h < hands.size(); h++) {
            HandLandmarks hand = hands.get(h);
            sb.append("    {\n");
            sb.append("      \"handIndex\": ").append(h).append(",\n");
            sb.append("      \"handednessScore\": ").append(hand.handednessScore).append(",\n");
            sb.append("      \"presenceScore\": ").append(hand.presenceScore).append(",\n");
            appendVector(sb, "      ", "imuNormal", h < imuNormal.length ? imuNormal[h] : null, true);
            appendVector(sb, "      ", "geometricNormal", h < geometricNormal.length ? geometricNormal[h] : null, true);
            sb.append("      \"landmarks\": [\n");
            List<Point3D> pts = hand.absolutePoints;
            for (int i = 0; i < pts.size(); i++) {
                Point3D p = pts.get(i);
                sb.append("        { \"index\": ").append(i)
                        .append(", \"x\": ").append(p.getX())
                        .append(", \"y\": ").append(p.getY())
                        .append(", \"z\": ").append(p.getZ())
                        .append(" }").append(i < pts.size() - 1 ? "," : "").append("\n");
            }
            sb.append("      ]\n");
            sb.append("    }").append(h < hands.size() - 1 ? "," : "").append("\n");
        }

        sb.append("  ]\n");
        sb.append("}\n");

        Files.writeString(path, sb.toString());
    }

    /** Appends {@code "name": {"x":..,"y":..,"z":..}} (or {@code null}) at the given indent. */
    private static void appendVector(StringBuilder sb, String indent, String name, Point3D v, boolean trailingComma) {
        sb.append(indent).append("\"").append(name).append("\": ");
        if (v == null) {
            sb.append("null");
        } else {
            sb.append("{ \"x\": ").append(v.getX())
                    .append(", \"y\": ").append(v.getY())
                    .append(", \"z\": ").append(v.getZ())
                    .append(" }");
        }
        sb.append(trailingComma ? ",\n" : "\n");
    }
}

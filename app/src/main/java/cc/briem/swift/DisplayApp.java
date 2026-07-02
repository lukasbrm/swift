package cc.briem.swift;

import java.util.List;

import cc.briem.swift.cv.models.HandLandmarks;
import cc.briem.swift.cv.models.LandmarkResult;

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

    private static final double DOT_RADIUS = 5.0;
    private static final double LINE_WIDTH = 2.0;
    private static final Color DOT_COLOR = Color.LIME;
    private static final Color LINE_COLOR = Color.WHITE;

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
    public static void render(Image image, LandmarkResult result) {
        if (imageView != null) imageView.setImage(image);
        drawOverlay(image, result);
    }

    private static void drawOverlay(Image image, LandmarkResult result) {
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
        }
    }
}

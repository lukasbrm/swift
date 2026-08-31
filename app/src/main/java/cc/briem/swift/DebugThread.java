package cc.briem.swift;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.imu.IMUController;
import cc.briem.swift.imu.ImuPacket;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.stage.Stage;

public class DebugThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(DebugThread.class);

    private static final String DEFAULT_PORT = "/dev/tty.usbserial-110";
    private static final int    DEFAULT_BAUD = 115200;

    private volatile Label accelLabel;
    private volatile Label angleLabel;

    @Override
    public void run() {
        logger.info("DebugThread starting..");

        CountDownLatch uiReady = new CountDownLatch(1);
        Platform.startup(() -> {
            buildWindow();
            uiReady.countDown();
        });

        try {
            uiReady.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;
        }

        IMUController imuController = new IMUController(DEFAULT_PORT, DEFAULT_BAUD);
        try {
            imuController.open();
            imuController.readLoop(this::onPacket);
        } catch (IOException e) {
            logger.error("IMU connection failed: {}", e.getMessage());
        } finally {
            imuController.close();
        }
    }

    private void buildWindow() {
        accelLabel = new Label("Acceleration: --");
        angleLabel = new Label("Angle: --");
        accelLabel.setFont(Font.font(16));
        angleLabel.setFont(Font.font(16));

        VBox root = new VBox(12, accelLabel, angleLabel);
        root.setPadding(new Insets(16));

        Stage stage = new Stage();
        stage.setTitle("IMU Debug");
        stage.setScene(new Scene(root, 320, 120));
        stage.show();
    }

    private void onPacket(ImuPacket packet) {
        switch (packet) {
            case ImuPacket.Acceleration a -> {
                String text = String.format(
                        "Acceleration: x=%sg, y=%sg, z=%sg",
                        format(a.ax()), format(a.ay()), format(a.az()));
                Platform.runLater(() -> accelLabel.setText(text));
            }
            case ImuPacket.Quaternion q -> {
                double[] rpy = eulerDegrees(q);
                String text = String.format(
                        "Angle: roll=%s°, pitch=%s°, yaw=%s°",
                        format(rpy[0]), format(rpy[1]), format(rpy[2]));
                Platform.runLater(() -> angleLabel.setText(text));
            }
            default -> { }
        }
    }

    /**
     * Converts a quaternion to roll/pitch/yaw degrees, display-only. The rest of the app works
     * with the quaternion directly (see {@code AnalysisThread.worldRotationMatrix}) specifically
     * to avoid this conversion's gimbal-lock singularity at pitch = ±90° — fine here since a
     * debug label glitching briefly at that one orientation doesn't feed into any tracking math.
     */
    private static double[] eulerDegrees(ImuPacket.Quaternion q) {
        double qw = q.qw(), qx = q.qx(), qy = q.qy(), qz = q.qz();

        double roll = Math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));

        double sinPitch = 2.0 * (qw * qy - qz * qx);
        sinPitch = Math.max(-1.0, Math.min(1.0, sinPitch));  // clamp against rounding error
        double pitch = Math.asin(sinPitch);

        double yaw = Math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));

        return new double[]{ Math.toDegrees(roll), Math.toDegrees(pitch), Math.toDegrees(yaw) };
    }

    private static String format(double value) {
        return String.format("%.2f", value);
    }
}

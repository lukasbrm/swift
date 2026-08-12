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

    private static final String DEFAULT_PORT = "/dev/tty.usbserial-210";
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
            case ImuPacket.Angle ang -> {
                String text = String.format(
                        "Angle: roll=%s°, pitch=%s°, yaw=%s°",
                        format(ang.roll()), format(ang.pitch()), format(ang.yaw()));
                Platform.runLater(() -> angleLabel.setText(text));
            }
            default -> { }
        }
    }

    private static String format(double value) {
        return String.format("%.2f", value);
    }
}

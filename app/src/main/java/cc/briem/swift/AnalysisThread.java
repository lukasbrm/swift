package cc.briem.swift;

import java.io.ByteArrayInputStream;
import java.time.Instant;
import java.util.concurrent.BlockingQueue;

import cc.briem.swift.imu.IMUController;
import cc.briem.swift.imu.ImuPacket;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.cv.models.LandmarkResult;
import cc.briem.swift.network.models.Frame;

import javafx.application.Platform;
import javafx.scene.image.Image;

public class AnalysisThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(AnalysisThread.class);

    private final BlockingQueue<Frame> analyzedFrames;
    private final BlockingQueue<LandmarkResult> landmarkResults;
    private final IMUController imuController;

    public AnalysisThread(BlockingQueue<Frame> analyzedFrames, BlockingQueue<LandmarkResult> landmarkResults, IMUController imuController) {
        this.analyzedFrames = analyzedFrames;
        this.landmarkResults = landmarkResults;
        this.imuController = imuController;
    }

    @Override
    public void run() {
        logger.info("AnalysisThread starting...");

        while (!Thread.currentThread().isInterrupted()) {
            try {
                // Take frame
                Frame frame = analyzedFrames.take();
                analyzedFrames.clear(); // drop backlog, always display the latest frame

                Image image = new Image(new ByteArrayInputStream(frame.getJpegData()));

                // Predict landmarks from frame
                LandmarkResult landmarkResult = landmarkResults.poll();
                landmarkResults.clear(); // keep in sync with frame backlog drop above

                // Get IMU angle data at frame time
                ImuPacket.Angle anglePacket = imuController.getImuAt(Instant.now(), ImuPacket.Angle.class).get();


                Platform.runLater(() -> DisplayApp.render(image, landmarkResult, anglePacket));

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.info("AnalysisThread interrupted, shutting down.");
            }
        }
    }
}

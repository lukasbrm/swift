package cc.briem.swift;

import java.io.ByteArrayInputStream;
import java.util.concurrent.BlockingQueue;

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

    public AnalysisThread(BlockingQueue<Frame> analyzedFrames, BlockingQueue<LandmarkResult> landmarkResults) {
        this.analyzedFrames = analyzedFrames;
        this.landmarkResults = landmarkResults;
    }

    @Override
    public void run() {
        logger.info("AnalysisThread starting...");

        while (!Thread.currentThread().isInterrupted()) {
            try {
                Frame frame = analyzedFrames.take();
                analyzedFrames.clear(); // drop backlog, always display the latest frame

                Image image = new Image(new ByteArrayInputStream(frame.getJpegData()));

                LandmarkResult result = landmarkResults.poll();
                landmarkResults.clear(); // keep in sync with frame backlog drop above

                Platform.runLater(() -> DisplayApp.render(image, result));

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.info("AnalysisThread interrupted, shutting down.");
            }
        }
    }
}

package cc.briem.swift;

import java.io.ByteArrayInputStream;
import java.util.concurrent.BlockingQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.network.models.Frame;
import javafx.application.Platform;
import javafx.scene.image.Image;

public class AnalysisThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(AnalysisThread.class);

    private final BlockingQueue<Frame> frameBuffer;

    public AnalysisThread(BlockingQueue<Frame> frameBuffer) {
        this.frameBuffer = frameBuffer;
    }

    @Override
    public void run() {

        logger.info("AnalysisThread starting...");

        while (!Thread.currentThread().isInterrupted()) {
            try {
                Frame frame = frameBuffer.take(); // blocks until a frame is available

                // Convert byte data to Image
                Image image = new Image(new ByteArrayInputStream(frame.getJpegData()));

                // Update the ImageView on the JavaFX Application Thread
                Platform.runLater(() -> {
                    if (DisplayApp.imageView != null) {
                        DisplayApp.imageView.setImage(image);
                    }
                });

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.info("AnalysisThread interrupted, shutting down.");
            }
        }
    }
}
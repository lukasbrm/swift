package cc.briem.swift;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.BlockingQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.onnxruntime.OrtException;
import cc.briem.swift.cv.HandLandmarkPipeline;
import cc.briem.swift.cv.models.LandmarkResult;
import cc.briem.swift.network.models.Frame;

public class CVThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(CVThread.class);

    private BlockingQueue<Frame> frameBuffer;
    private BlockingQueue<LandmarkResult> landmarkResults;
    private BlockingQueue<Frame> analyzedFrames;

    public CVThread(BlockingQueue<Frame> frameBuffer, BlockingQueue<LandmarkResult> landmarkResults, BlockingQueue<Frame> analyzedFrames) {
        this.frameBuffer = frameBuffer;
        this.landmarkResults = landmarkResults;
        this.analyzedFrames = analyzedFrames;
    }

    @Override
    public void run(){

        logger.info("CVThread starting...");

        try {

            // Extract hand landmarks
            String palmModel    = extractResource("/models/palm_detection_mediapipe_2023feb.onnx");
            String anchors      = extractResource("/models/anchors.csv");
            String landmarkModel = extractResource("/models/handpose_estimation_mediapipe_2023feb.onnx");
            try (HandLandmarkPipeline pipeline = new HandLandmarkPipeline(palmModel, anchors, landmarkModel)) {
                while (!Thread.currentThread().isInterrupted()) {
                Frame frame = frameBuffer.take();
                frameBuffer.clear();
                LandmarkResult result = pipeline.process(frame);
                landmarkResults.offer(result);
                analyzedFrames.offer(frame);
                logger.debug(result.toString());
                }
            }

        } catch(InterruptedException e) {
            logger.error("Frame could not be fetched from frame buffer");
            throw new RuntimeException("Frame could not be fetched from frame buffer", e);
        } catch(OrtException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String extractResource(String resourcePath) throws IOException {
        try (InputStream in = CVThread.class.getResourceAsStream(resourcePath)) {
            Path tmp = Files.createTempFile("swift-", Paths.get(resourcePath).getFileName().toString());
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            return tmp.toAbsolutePath().toString();
        }
    }
}
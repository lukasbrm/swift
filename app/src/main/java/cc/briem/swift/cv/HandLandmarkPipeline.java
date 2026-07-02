package cc.briem.swift.cv;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OrtException;
import cc.briem.swift.cv.models.HandLandmarks;
import cc.briem.swift.cv.models.LandmarkResult;
import cc.briem.swift.network.models.Frame;

public class HandLandmarkPipeline implements AutoCloseable {

    private final PalmDetector palmDetector;
    private final LandmarkPredictor landmarkPredictor;

    public HandLandmarkPipeline(String palmModelPath, String anchorsCsvPath, String landmarkModelPath) throws OrtException, IOException {
        this.palmDetector = new PalmDetector(palmModelPath, anchorsCsvPath);
        this.landmarkPredictor = new LandmarkPredictor(landmarkModelPath);
    }

    public LandmarkResult process(Frame frame) throws OrtException, IOException{
        
        // First detect palms
        List<PalmDetector.Detection> detections = palmDetector.detect(frame.toBufferedImage());
        List<HandLandmarks> hands = new ArrayList<>();

        // For each detected palm, predict landmarks
        for(PalmDetector.Detection detection : detections) {
            HandLandmarks landmarks = landmarkPredictor.predict(frame.toBufferedImage(), detection);
            if(landmarks != null) {
                hands.add(landmarks);
            }
        }
        return new LandmarkResult(hands);
    }

    @Override
    public void close() throws OrtException {
        palmDetector.close();
        landmarkPredictor.close();
    }
}
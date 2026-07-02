package cc.briem.swift.cv.models;

import java.util.List;

public class LandmarkResult {

    private final List<HandLandmarks> hands;
    private final double timestamp;

    public LandmarkResult(List<HandLandmarks> hands) {
        this.hands = hands;
        this.timestamp = System.currentTimeMillis();
    }

    public List<HandLandmarks> getHands() {
        return hands;
    }

    public double getTimestamp() {
        return timestamp;
    }

    @Override
    public String toString() {
        return "Hands=" + hands.size() + " Timestamp=" + timestamp;
    }

}
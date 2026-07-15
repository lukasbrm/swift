package cc.briem.swift.cv.models;

import java.time.Instant;
import java.util.List;

public class LandmarkResult {

    private final List<HandLandmarks> hands;
    private final Instant timestamp;

    public LandmarkResult(List<HandLandmarks> hands) {
        this.hands = hands;
        this.timestamp = Instant.now();
    }

    public List<HandLandmarks> getHands() {
        return hands;
    }

    public void setFirstHandLandmarks(HandLandmarks landmarks) {
        hands.set(0, landmarks);
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    @Override
    public String toString() {
        return "Hands=" + hands.size() + " Timestamp=" + timestamp;
    }

}
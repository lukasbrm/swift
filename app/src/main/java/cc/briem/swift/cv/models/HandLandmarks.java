package cc.briem.swift.cv.models;

import java.util.ArrayList;
import java.util.List;

import javafx.geometry.Point3D;

public class HandLandmarks {
    public final List<Point3D> points;          // 21 (x,y,z) in image coordinates
    public final List<Point3D> worldPoints;     // 21 (x,y,z) in metric world coordinates (MediaPipe Identity_3)
    public final double handednessScore; // >0.5 -> right else left
    public final double presenceScore;

    public HandLandmarks(List<Point3D> points, List<Point3D> worldPoints, double handednessScore, double presenceScore) {
        this.points = points;
        this.worldPoints = worldPoints;
        this.handednessScore = handednessScore;
        this.presenceScore = presenceScore;
    }

    public HandLandmarks add(HandLandmarks other) {
        List<Point3D> newPoints = new ArrayList<>(this.points.size());
        for(int i = 0; i < this.points.size(); i++) {
            newPoints.add(i, this.points.get(i).add(other.points.get(i)));
        }

        return new HandLandmarks(newPoints, worldPoints, handednessScore, presenceScore);
    }

    public HandLandmarks subtract(HandLandmarks other) {
        List<Point3D> newPoints = new ArrayList<>(this.points.size());
        for(int i = 0; i < this.points.size(); i++) {
            newPoints.add(i, this.points.get(i).subtract(other.points.get(i)));
        }

        return new HandLandmarks(newPoints, worldPoints, handednessScore, presenceScore);
    }

    public HandLandmarks multiply(double other) {
        List<Point3D> newPoints = new ArrayList<>(this.points.size());
        for(int i = 0; i < this.points.size(); i++) {
            newPoints.add(i, this.points.get(i).multiply(other));
        }
        return new HandLandmarks(newPoints, worldPoints, handednessScore, presenceScore);
    }

    public HandLandmarks divide(double other) {
        return this.multiply(1.0 / other);
    }
}
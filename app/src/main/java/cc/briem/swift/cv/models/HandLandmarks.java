package cc.briem.swift.cv.models;

import java.util.List;

import javafx.geometry.Point3D;

public class HandLandmarks {
    public final List<Point3D> points;
    public final double handednessScore; // >0.5 -> right else left
    public final double presenceScore;

    public HandLandmarks(List<Point3D> points, double handednessScore, double presenceScore) {
        this.points = points;
        this.handednessScore = handednessScore;
        this.presenceScore = presenceScore;
    }
}
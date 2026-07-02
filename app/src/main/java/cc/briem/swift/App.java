package cc.briem.swift;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import cc.briem.swift.network.models.Frame;
import javafx.application.Application;

public class App {
    public static void main(String[] args) {

        // Instantiate frame buffer
        BlockingQueue<Frame> frameBuffer = new LinkedBlockingQueue<>(641);

        // Start network thread (receives frames)
        NetworkThread networkThreadRunnable = new NetworkThread(frameBuffer);
        Thread networkThread = new Thread(networkThreadRunnable);
        networkThread.setDaemon(true);
        networkThread.start();

        // Start CV thread
        CVThread cvThreadRunnable = new CVThread(frameBuffer);
        Thread cvThread = new Thread(cvThreadRunnable);
        cvThread.setDaemon(true);
        cvThread.start();

        // Start IMU thread
        IMUThread imuThreadRunnable = new IMUThread(frameBuffer);
        Thread imuThread = new Thread(imuThreadRunnable);
        imuThread.setDaemon(true);
        imuThread.start();

        // Start Analysis thread (consumes frames and updates the JavaFX UI)
        AnalysisThread analysisThreadRunnable = new AnalysisThread(frameBuffer);
        Thread analysisThread = new Thread(analysisThreadRunnable);
        analysisThread.setDaemon(true);
        analysisThread.start();

        // Launch the JavaFX display window (blocks until the window is closed)
        Application.launch(DisplayApp.class, args);
    }
}

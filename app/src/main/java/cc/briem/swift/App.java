package cc.briem.swift;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import cc.briem.swift.cv.models.LandmarkResult;
import cc.briem.swift.imu.IMUController;
import cc.briem.swift.network.models.Frame;
import javafx.application.Application;

public class App {
    public static void main(String[] args) {

        nu.pattern.OpenCV.loadLocally();

        if(false) {
            DebugThread debugThread = new DebugThread();
            Thread thread = new Thread(debugThread);
            thread.start();
        } else {

            // Instantiate frame buffer and Landmark buffer
            BlockingQueue<Frame> frameBuffer = new LinkedBlockingQueue<>(64);
            BlockingQueue<Frame> analyzedFrames = new LinkedBlockingQueue<>(64);
            BlockingQueue<LandmarkResult> landmarkResults = new LinkedBlockingQueue<>(64);

            // Instantiate IMUController for global access
            IMUController imuController = new IMUController("/dev/tty.usbserial-210", 115200);

            // Start network thread (receives frames)
            NetworkThread networkThreadRunnable = new NetworkThread(frameBuffer);
            Thread networkThread = new Thread(networkThreadRunnable, "Network Thread");
            networkThread.setDaemon(true);
            networkThread.start();

            // Start CV thread
            CVThread cvThreadRunnable = new CVThread(frameBuffer, landmarkResults, analyzedFrames);
            Thread cvThread = new Thread(cvThreadRunnable, "CV Thread");
            cvThread.setDaemon(true);
            cvThread.start();

            // Start IMU thread
            IMUThread imuThreadRunnable = new IMUThread(imuController);
            Thread imuThread = new Thread(imuThreadRunnable, "IMU Thread");
            imuThread.setDaemon(true);
            imuThread.start();

            // Start Analysis thread (consumes frames and updates the JavaFX UI)
            AnalysisThread analysisThreadRunnable = new AnalysisThread(analyzedFrames, landmarkResults, imuController);
            Thread analysisThread = new Thread(analysisThreadRunnable, "Analysis Thread");
            analysisThread.setDaemon(true);
            analysisThread.start();

            // Launch the JavaFX display window (blocks until the window is closed)
            Application.launch(DisplayApp.class, args);
        }
    }
}

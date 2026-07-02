package cc.briem.swift;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.imu.IMUController;

public class IMUThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(IMUThread.class);

    private static final String DEFAULT_PORT = "/dev/tty.usbserial-210";
    private static final int    DEFAULT_BAUD = 115200;

    private final IMUController imuController;

    public IMUThread(IMUController imuController) {
        this.imuController = imuController;
    }

    /** Exposes the controller so other threads can call {@code getImuAt}. */
    public IMUController getController() {
        return imuController;
    }

    @Override
    public void run() {
        logger.info("IMUThread starting on port {} ...", DEFAULT_PORT);
        try {
            imuController.open();
            imuController.readLoop(p -> {});  // blocks this thread; history is maintained internally
        } catch (IOException e) {
            logger.error("IMU connection failed: {}", e.getMessage());
        } finally {
            imuController.close();
        }
    }
}

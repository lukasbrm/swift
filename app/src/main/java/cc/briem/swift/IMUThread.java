package cc.briem.swift;

import java.util.concurrent.BlockingQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.network.models.Frame;

public class IMUThread implements Runnable {
    
    private static final Logger logger = LoggerFactory.getLogger(IMUThread.class);

    private BlockingQueue<Frame> frameBuffer;

    public IMUThread(BlockingQueue<Frame> frameBuffer) {
        this.frameBuffer = frameBuffer;
    }

    @Override
    public void run() {

        logger.info("IMUThread starting...");

    }
}
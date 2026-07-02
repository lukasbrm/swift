package cc.briem.swift;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.BlockingQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.briem.swift.network.models.Frame;

public class NetworkThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(IMUThread.class);
    private static final int PORT = 4711;
    private static final int HEADER_SIZE = 8;
    private static final int MAX_PACKET_SIZE = 65536;

    private volatile boolean running = true;

    // State for the frame currently assembling
    private int currentTotalSize = -1;
    private int currentNumChunks = -1;
    private byte[][] chunkBuffer;
    private int chunksReceived = 0;

    private BlockingQueue<Frame> frameBuffer;

    public NetworkThread(BlockingQueue<Frame> frameBuffer) {
        this.frameBuffer = frameBuffer;
    }

    @Override
    public void run() {

        logger.info("NetworkThread starting...");

        try(DatagramSocket socket = new DatagramSocket(PORT)) {
            socket.setReceiveBufferSize(1024 * 1024);

            byte[] recvBuf = new byte[MAX_PACKET_SIZE];
            DatagramPacket packet = new DatagramPacket(recvBuf, recvBuf.length);

            while(running) {
                socket.receive(packet);
                handlePacket(packet.getData(), packet.getLength());
                //logger.debug("Frame Buffer size: " + frameBuffer.size());
            }

        } catch(SocketException e) {
            if(running) {
                logger.error("UDP socket error");
                throw new RuntimeException("UDP socket error", e);
            }
        } catch(Exception e) {
            logger.error("Error receiving frames");
            throw new RuntimeException("Error receiving frames", e);
        }
    }

    private void handlePacket(byte[] data, int length) {
        if (length < HEADER_SIZE) {
            return; // malformed, ignore
        }
 
        ByteBuffer buf = ByteBuffer.wrap(data, 0, HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
        int chunkIndex = buf.getShort() & 0xFFFF;
        int numChunks = buf.getShort() & 0xFFFF;
        int totalSize = buf.getInt(); // fits in signed int for realistic JPEG sizes
 
        int payloadLen = length - HEADER_SIZE;
 
        // New frame starting (chunkIndex 0) -> reset assembly state.
        // This also naturally recovers from a previous frame that lost packets.
        if (chunkIndex == 0) {
            currentTotalSize = totalSize;
            currentNumChunks = numChunks;
            chunkBuffer = new byte[numChunks][];
            chunksReceived = 0;
        }
 
        // Ignore stray/late chunks that don't belong to the frame we're assembling
        // (e.g. arrived after we already reset, or from a dropped-then-restarted frame).
        if (chunkBuffer == null || numChunks != currentNumChunks || totalSize != currentTotalSize) {
            return;
        }
        if (chunkIndex < 0 || chunkIndex >= chunkBuffer.length) {
            return;
        }
        if (chunkBuffer[chunkIndex] != null) {
            return; // duplicate
        }
 
        byte[] chunkData = new byte[payloadLen];
        System.arraycopy(data, HEADER_SIZE, chunkData, 0, payloadLen);
        chunkBuffer[chunkIndex] = chunkData;
        chunksReceived++;
 
        if (chunksReceived == currentNumChunks) {
            assembleAndEmit();
        }
    }
 
    private void assembleAndEmit() {
        byte[] full = new byte[currentTotalSize];
        int offset = 0;
        for (byte[] chunk : chunkBuffer) {
            System.arraycopy(chunk, 0, full, offset, chunk.length);
            offset += chunk.length;
        }
 
        Frame frame = new Frame(full);
 
        // offer() instead of put() so a full queue never blocks the receive loop
        // and stalls UDP reads (which would just cause more packet loss).
        if (!frameBuffer.offer(frame)) {
            //System.err.println("Frame queue full, dropping frame");
        }
 
        // reset so a duplicate/late packet for this frame index can't be mistaken for a new one
        chunkBuffer = null;
        currentNumChunks = -1;
        currentTotalSize = -1;
        chunksReceived = 0;
    }
 
    public void stop() {
        running = false;
    }


    
}
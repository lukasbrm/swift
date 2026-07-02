package cc.briem.swift.imu;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.Deque;
import java.util.Optional;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.function.Consumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fazecast.jSerialComm.SerialPort;

/**
 * Controls a WIT-protocol IMU sensor over a serial (UART) connection.
 *
 * <p>The WIT protocol uses fixed 11-byte packets:
 * <pre>
 *   [0]  0x55        – start byte
 *   [1]  type        – 0x51 accel | 0x52 gyro | 0x53 angle | 0x54 magnetic
 *   [2–9] payload    – 4 × signed 16-bit little-endian values
 *   [10] checksum    – lower 8 bits of sum of bytes 0–9
 * </pre>
 *
 * <p>Usage:
 * <pre>{@code
 *   try (IMUController imu = new IMUController("/dev/tty.usbserial-210", 115200)) {
 *       imu.open();
 *       imu.readLoop(packet -> System.out.println(packet)); // blocks the calling thread
 *   }
 * }</pre>
 */
public class IMUController implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(IMUController.class);

    // WIT protocol constants
    private static final byte   START_BYTE    = 0x55;
    private static final byte   TYPE_ACCEL    = 0x51;
    private static final byte   TYPE_GYRO     = 0x52;
    private static final byte   TYPE_ANGLE    = 0x53;
    private static final byte   TYPE_MAGNETIC = 0x54;
    private static final int    PACKET_SIZE   = 11;

    // Scaling factors from the WIT datasheet
    private static final double ACCEL_SCALE   = 16.0   / 32768.0;  // g per LSB
    private static final double GYRO_SCALE    = 2000.0 / 32768.0;  // °/s per LSB
    private static final double ANGLE_SCALE   = 180.0  / 32768.0;  // ° per LSB
    private static final double TEMP_SCALE    = 1.0    / 100.0;    // °C per LSB

    private final String portName;
    private final int    baudRate;

    /** Maximum number of packets kept per type for temporal lookup. */
    private static final int HISTORY_SIZE = 256;

    // Per-type ring buffers — bounded, thread-safe
    private final Deque<ImuPacket.Acceleration> accelHistory = new ConcurrentLinkedDeque<>();
    private final Deque<ImuPacket.Gyro>         gyroHistory  = new ConcurrentLinkedDeque<>();
    private final Deque<ImuPacket.Angle>        angleHistory = new ConcurrentLinkedDeque<>();
    private final Deque<ImuPacket.Magnetic>     magHistory   = new ConcurrentLinkedDeque<>();

    private SerialPort serialPort;

    /**
     * Creates an IMUController for the given serial port.
     *
     * @param portName serial port identifier (e.g. {@code "/dev/tty.usbserial-210"} or {@code "COM3"})
     * @param baudRate baud rate (typically 115200)
     */
    public IMUController(String portName, int baudRate) {
        this.portName = portName;
        this.baudRate = baudRate;
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Opens the serial port.
     *
     * @throws IOException if the port cannot be opened
     */
    public void open() throws IOException {
        serialPort = SerialPort.getCommPort(portName);
        serialPort.setBaudRate(baudRate);
        serialPort.setNumDataBits(8);
        serialPort.setNumStopBits(SerialPort.ONE_STOP_BIT);
        serialPort.setParity(SerialPort.NO_PARITY);
        serialPort.setComPortTimeouts(SerialPort.TIMEOUT_READ_BLOCKING, 1000, 0);

        if (!serialPort.openPort()) {
            throw new IOException("Failed to open serial port: " + portName);
        }
        logger.info("Serial port opened: {} @ {} baud", portName, baudRate);
    }

    /**
     * Closes the serial port.
     */
    @Override
    public void close() {
        if (serialPort != null && serialPort.isOpen()) {
            serialPort.closePort();
            logger.info("Serial port closed: {}", portName);
        }
    }

    /**
     * Returns {@code true} if the underlying serial port is currently open.
     */
    public boolean isOpen() {
        return serialPort != null && serialPort.isOpen();
    }

    // -------------------------------------------------------------------------
    // Blocking read loop — runs on the caller's thread (e.g. IMUThread)
    // -------------------------------------------------------------------------

    /**
     * Blocks the calling thread, reading and parsing packets until the thread is interrupted.
     * Each successfully parsed packet is delivered to {@code packetConsumer} on the calling thread.
     *
     * <p>Intended to be called directly from {@code IMUThread.run()}.
     *
     * @param packetConsumer callback invoked for every successfully parsed packet
     * @throws IllegalStateException if the port has not been opened yet
     */
    public void readLoop(Consumer<ImuPacket> packetConsumer) {
        if (serialPort == null || !serialPort.isOpen()) {
            throw new IllegalStateException("Serial port is not open. Call open() first.");
        }

        byte[] buf  = new byte[PACKET_SIZE * 4];  // small sliding buffer
        int    fill = 0;

        InputStream in = serialPort.getInputStream();

        logger.info("IMU read loop running on port {}", portName);
        try {
            while (!Thread.currentThread().isInterrupted()) {
                int available = in.available();
                int toRead    = Math.max(1, Math.min(available, buf.length - fill));
                int read      = in.read(buf, fill, toRead);
                if (read <= 0) continue;
                fill += read;

                // Consume as many complete packets as possible from the front of the buffer
                int consumed = 0;
                while (fill - consumed >= PACKET_SIZE) {
                    int startIdx = findStartByte(buf, consumed, fill);
                    if (startIdx == -1) {
                        consumed = fill;  // no start byte found — discard everything
                        break;
                    }
                    consumed = startIdx;  // discard leading garbage before start byte

                    if (fill - consumed < PACKET_SIZE) break;  // need more data

                    parsePacket(buf, consumed).ifPresent(p -> {
                        store(p);
                        packetConsumer.accept(p);
                    });
                    consumed += PACKET_SIZE;
                }

                // Compact the buffer: shift unconsumed bytes to the front
                if (consumed > 0) {
                    int remaining = fill - consumed;
                    System.arraycopy(buf, consumed, buf, 0, remaining);
                    fill = remaining;
                }
            }
        } catch (IOException e) {
            logger.error("IMU read error on {}: {}", portName, e.getMessage());
        }
        logger.info("IMU read loop terminated.");
    }

    // -------------------------------------------------------------------------
    // Packet parsing
    // -------------------------------------------------------------------------

    /**
     * Attempts to parse a single 11-byte WIT packet starting at {@code buf[offset]}.
     *
     * @param buf    byte array containing at least {@code offset + 11} bytes
     * @param offset start index of the candidate packet
     * @return a parsed {@link ImuPacket}, or empty on checksum/format error
     */
    public Optional<ImuPacket> parsePacket(byte[] buf, int offset) {
        if (buf[offset] != START_BYTE) return Optional.empty();
        if (!isChecksumValid(buf, offset)) return Optional.empty();

        byte    type  = buf[offset + 1];
        Instant ts    = Instant.now();
        short[] words = extractWords(buf, offset + 2);  // 4 × int16

        return switch (type) {
            case TYPE_ACCEL    -> Optional.of(parseAcceleration(words, ts));
            case TYPE_GYRO     -> Optional.of(parseGyro(words, ts));
            case TYPE_ANGLE    -> Optional.of(parseAngle(words, ts));
            case TYPE_MAGNETIC -> Optional.of(parseMagnetic(words, ts));
            default -> {
                logger.debug("Unknown packet type: 0x{}", String.format("%02X", type));
                yield Optional.empty();
            }
        };
    }

    // -------------------------------------------------------------------------
    // Checksum & byte helpers
    // -------------------------------------------------------------------------

    /** Validates the WIT checksum: {@code sum(bytes[0..9]) & 0xFF == byte[10]}. */
    private static boolean isChecksumValid(byte[] buf, int offset) {
        int sum = 0;
        for (int i = offset; i < offset + PACKET_SIZE - 1; i++) {
            sum += buf[i] & 0xFF;
        }
        return (sum & 0xFF) == (buf[offset + PACKET_SIZE - 1] & 0xFF);
    }

    /** Returns the index of the first {@code 0x55} byte in {@code buf[from..to)}, or -1. */
    private static int findStartByte(byte[] buf, int from, int to) {
        for (int i = from; i < to; i++) {
            if (buf[i] == START_BYTE) return i;
        }
        return -1;
    }

    /** Reads four signed 16-bit little-endian values from {@code buf[offset]}. */
    private static short[] extractWords(byte[] buf, int offset) {
        ByteBuffer bb = ByteBuffer.wrap(buf, offset, 8).order(ByteOrder.LITTLE_ENDIAN);
        return new short[]{ bb.getShort(), bb.getShort(), bb.getShort(), bb.getShort() };
    }

    // -------------------------------------------------------------------------
    // Temporal lookup
    // -------------------------------------------------------------------------

    /**
     * Returns the packet of the requested type whose timestamp is closest to {@code target}.
     * Safe to call from any thread.
     *
     * @param target     the reference timestamp to search near
     * @param packetType the desired subtype (e.g. {@code ImuPacket.Acceleration.class})
     * @return the closest matching packet, or empty if no packet of that type has been received yet
     */
    public <T extends ImuPacket> Optional<T> getImuAt(Instant target, Class<T> packetType) {
        Deque<? extends T> history = historyFor(packetType);
        if (history == null) return Optional.empty();

        @SuppressWarnings("unchecked")
        Optional<T> result = (Optional<T>) history.stream()
                .min(Comparator.comparingLong(p -> absDiffNanos(p.timestamp(), target)));
        return result;
    }

    // -------------------------------------------------------------------------
    // History helpers
    // -------------------------------------------------------------------------

    private void store(ImuPacket packet) {
        switch (packet) {
            case ImuPacket.Acceleration a -> append(accelHistory, a);
            case ImuPacket.Gyro         g -> append(gyroHistory,  g);
            case ImuPacket.Angle      ang -> append(angleHistory, ang);
            case ImuPacket.Magnetic     m -> append(magHistory,   m);
        }
    }

    private static <T> void append(Deque<T> deque, T item) {
        deque.addLast(item);
        while (deque.size() > HISTORY_SIZE) {
            deque.pollFirst();
        }
    }

    @SuppressWarnings("unchecked")
    private <T extends ImuPacket> Deque<T> historyFor(Class<T> type) {
        if (type == ImuPacket.Acceleration.class) return (Deque<T>) accelHistory;
        if (type == ImuPacket.Gyro.class)         return (Deque<T>) gyroHistory;
        if (type == ImuPacket.Angle.class)        return (Deque<T>) angleHistory;
        if (type == ImuPacket.Magnetic.class)     return (Deque<T>) magHistory;
        return null;
    }

    private static long absDiffNanos(Instant a, Instant b) {
        return Math.abs(Duration.between(a, b).toNanos());
    }

    // -------------------------------------------------------------------------
    // Type-specific parsers
    // -------------------------------------------------------------------------

    private static ImuPacket.Acceleration parseAcceleration(short[] w, Instant ts) {
        return new ImuPacket.Acceleration(
                w[0] * ACCEL_SCALE,
                w[1] * ACCEL_SCALE,
                w[2] * ACCEL_SCALE,
                w[3] * TEMP_SCALE,
                ts);
    }

    private static ImuPacket.Gyro parseGyro(short[] w, Instant ts) {
        return new ImuPacket.Gyro(
                w[0] * GYRO_SCALE,
                w[1] * GYRO_SCALE,
                w[2] * GYRO_SCALE,
                w[3] * TEMP_SCALE,
                ts);
    }

    private static ImuPacket.Angle parseAngle(short[] w, Instant ts) {
        return new ImuPacket.Angle(
                w[0] * ANGLE_SCALE,
                w[1] * ANGLE_SCALE,
                w[2] * ANGLE_SCALE,
                w[3] * TEMP_SCALE,
                ts);
    }

    private static ImuPacket.Magnetic parseMagnetic(short[] w, Instant ts) {
        return new ImuPacket.Magnetic(
                w[0], w[1], w[2],
                w[3] * TEMP_SCALE,
                ts);
    }
}

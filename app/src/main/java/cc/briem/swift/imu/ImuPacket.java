package cc.briem.swift.imu;

import java.time.Instant;

/**
 * Sealed interface representing a parsed WIT-protocol IMU packet (11 bytes).
 * Each subtype corresponds to one packet type (0x51–0x54).
 */
public sealed interface ImuPacket permits
        ImuPacket.Acceleration,
        ImuPacket.Gyro,
        ImuPacket.Angle,
        ImuPacket.Magnetic {

    /** Timestamp when the packet was received. */
    Instant timestamp();

    /** Temperature in °C, included in every packet type. */
    double tempCelsius();

    /**
     * Packet type 0x51 — linear acceleration.
     *
     * @param ax         X-axis acceleration in g
     * @param ay         Y-axis acceleration in g
     * @param az         Z-axis acceleration in g
     * @param tempCelsius temperature in °C
     * @param timestamp  receive timestamp
     */
    record Acceleration(
            double ax,
            double ay,
            double az,
            double tempCelsius,
            Instant timestamp
    ) implements ImuPacket {}

    /**
     * Packet type 0x52 — angular rate (gyroscope).
     *
     * @param gx         X-axis angular rate in °/s
     * @param gy         Y-axis angular rate in °/s
     * @param gz         Z-axis angular rate in °/s
     * @param tempCelsius temperature in °C
     * @param timestamp  receive timestamp
     */
    record Gyro(
            double gx,
            double gy,
            double gz,
            double tempCelsius,
            Instant timestamp
    ) implements ImuPacket {}

    /**
     * Packet type 0x53 — Euler angles.
     *
     * @param roll        roll angle in degrees
     * @param pitch       pitch angle in degrees
     * @param yaw         yaw angle in degrees
     * @param tempCelsius temperature in °C
     * @param timestamp   receive timestamp
     */
    record Angle(
            double roll,
            double pitch,
            double yaw,
            double tempCelsius,
            Instant timestamp
    ) implements ImuPacket {}

    /**
     * Packet type 0x54 — raw magnetometer counts.
     *
     * @param hx          X-axis raw count
     * @param hy          Y-axis raw count
     * @param hz          Z-axis raw count
     * @param tempCelsius temperature in °C
     * @param timestamp   receive timestamp
     */
    record Magnetic(
            int hx,
            int hy,
            int hz,
            double tempCelsius,
            Instant timestamp
    ) implements ImuPacket {}
}

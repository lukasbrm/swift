package cc.briem.swift.imu;

import java.time.Instant;

/**
 * Sealed interface representing a parsed WIT-protocol IMU packet (11 bytes).
 * Each subtype corresponds to one packet type (0x51–0x54).
 */
public sealed interface ImuPacket permits
        ImuPacket.Acceleration,
        ImuPacket.Gyro,
        ImuPacket.Quaternion,
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
     * Packet type 0x59 — orientation, as a unit quaternion (Hamilton convention, body-to-world).
     *
     * <p>Used directly for rotation math (see {@code AnalysisThread.worldRotationMatrix}) instead
     * of converting to Euler roll/pitch/yaw: a quaternion has no gimbal-lock singularity, which
     * Euler angles do at pitch = ±90° — a pose a wrist-mounted sensor reaches easily (forearm
     * held vertical).
     *
     * @param qw          scalar component
     * @param qx          X component
     * @param qy          Y component
     * @param qz          Z component
     * @param tempCelsius temperature in °C
     * @param timestamp   receive timestamp
     */
    record Quaternion(
            double qw,
            double qx,
            double qy,
            double qz,
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

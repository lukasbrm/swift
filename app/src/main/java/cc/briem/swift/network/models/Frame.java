package cc.briem.swift.network.models;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Frame {
    
    private final byte[] jpegData;
    private final long timestamp;

    public Frame(byte[] jpegData) {
        this.jpegData = jpegData;
        this.timestamp = System.currentTimeMillis();
    }

    public byte[] getJpegData() {
        return jpegData;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public BufferedImage toBufferedImage() throws IOException {
        return ImageIO.read(new ByteArrayInputStream(jpegData));
    }

    @Override
    public String toString() {
        return "Frame{bytes=" + jpegData.length + ", ts=" + timestamp + "}";
    }
}
package cc.briem.swift.cv;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

public class ImageUtil {
    public static BufferedImage resize(BufferedImage src, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    public static BufferedImage crop(BufferedImage src, java.awt.Rectangle rect) {
        return src.getSubimage(rect.x, rect.y, rect.width, rect.height);
    }
}
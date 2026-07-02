package cc.briem.swift.cv;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import cc.briem.swift.cv.models.HandLandmarks;
import javafx.geometry.Point3D;

/**
 * Hand landmark model matching opencv/handpose_estimation_mediapipe's
 * mp_handpose.py (the "MPHandPose" class). Ported faithfully, including the
 * rotate-crop-to-upright preprocessing step.
 *
 * IMPORTANT: verify against your .onnx file in Netron:
 *  - output tensor names/order. This port assumes 4 outputs in this order:
 *    landmarks (1,63), conf (1,1), handedness (1,1), landmarks_word (1,63).
 *    Some conversions may name/order these differently.
 *  - conf/handedness here are used RAW (no sigmoid) per the reference code -
 *    this differs from PalmDetector's score output, which does need sigmoid.
 */
public class LandmarkPredictor implements AutoCloseable {

    private static final int INPUT_SIZE = 224;
    private static final String INPUT_NAME = "input_1";
    private static final String LANDMARKS_OUTPUT = "Identity";
    private static final String CONF_OUTPUT = "Identity_1";
    private static final String HANDEDNESS_OUTPUT = "Identity_2";
    private static final String LANDMARKS_WORLD_OUTPUT = "Identity_3";

    private static final float CONF_THRESHOLD = 0.8f;

    // Palm bbox pre-processing constants (from mp_handpose.py)
    private static final double[] PRE_SHIFT = {0, 0};
    private static final double PRE_ENLARGE = 4;
    private static final double[] SHIFT = {0, -0.4};
    private static final double ENLARGE = 3;

    private final OrtEnvironment env;
    private final OrtSession session;

    public LandmarkPredictor(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public HandLandmarks predict(BufferedImage frame, PalmDetector.Detection detection) throws OrtException {
        double[] box = {detection.xmin, detection.ymin, detection.xmax, detection.ymax};

        // Step A: crop+pad around the palm box (for rotation), enlarge x4
        CropResult step1 = cropAndPadFromPalm(frame, box, true);

        // Step B: compute rotation angle from 2 palm landmarks, rotate the crop upright
        double[] padBias = step1.bias.clone();
        double[] palmBoxLocal = subtract(step1.box, padBias);
        AffineUtil.Point2D[] palmLandmarksLocal = new AffineUtil.Point2D[7];
        for (int i = 0; i < 7; i++) {
            palmLandmarksLocal[i] = new AffineUtil.Point2D(
                detection.landmarks[i].x - padBias[0],
                detection.landmarks[i].y - padBias[1]);
        }

        AffineUtil.Point2D p1 = palmLandmarksLocal[0]; // wrist / palm base
        AffineUtil.Point2D p2 = palmLandmarksLocal[2]; // middle finger base
        double radians = Math.PI / 2 - Math.atan2(-(p2.y - p1.y), p2.x - p1.x);
        radians -= 2 * Math.PI * Math.floor((radians + Math.PI) / (2 * Math.PI));
        double angleDeg = Math.toDegrees(radians);

        double centerX = (palmBoxLocal[0] + palmBoxLocal[2]) / 2;
        double centerY = (palmBoxLocal[1] + palmBoxLocal[3]) / 2;
        AffineUtil.Mat2x3 rotMatrix = AffineUtil.rotationMatrix2D(centerX, centerY, angleDeg);

        BufferedImage rotatedImage = AffineUtil.warpAffine(step1.image, rotMatrix);

        double[] rotXs = new double[7], rotYs = new double[7];
        for (int i = 0; i < 7; i++) {
            AffineUtil.Point2D rp = AffineUtil.applyForward(rotMatrix, palmLandmarksLocal[i].x, palmLandmarksLocal[i].y);
            rotXs[i] = rp.x; rotYs[i] = rp.y;
        }
        double[] rotatedPalmBbox = {min(rotXs), min(rotYs), max(rotXs), max(rotYs)};

        // Step C: crop+pad again around the rotated palm box (final crop for the model), enlarge x3
        CropResult step2 = cropAndPadFromPalm(rotatedImage, rotatedPalmBbox, false);

        BufferedImage resized = ImageUtil.resize(step2.image, INPUT_SIZE, INPUT_SIZE);
        float[] inputData = preprocess(resized);
        long[] shape = {1, INPUT_SIZE, INPUT_SIZE, 3};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(INPUT_NAME, inputTensor);
            try (OrtSession.Result result = session.run(inputs)) {
                float[] rawLandmarks = ((float[][]) result.get(LANDMARKS_OUTPUT).get().getValue())[0];
                float conf = ((float[][]) result.get(CONF_OUTPUT).get().getValue())[0][0];
                float handedness = ((float[][]) result.get(HANDEDNESS_OUTPUT).get().getValue())[0][0];

                if (conf < CONF_THRESHOLD) return null;

                return postprocess(rawLandmarks, conf, handedness, step2.box, angleDeg, rotMatrix, step1.bias);
            }
        }
    }

    private float[] preprocess(BufferedImage image) {
        float[] data = new float[INPUT_SIZE * INPUT_SIZE * 3];
        int idx = 0;
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                int rgb = image.getRGB(x, y);
                data[idx++] = ((rgb >> 16) & 0xFF) / 255f;
                data[idx++] = ((rgb >> 8) & 0xFF) / 255f;
                data[idx++] = (rgb & 0xFF) / 255f;
            }
        }
        return data;
    }

    private HandLandmarks postprocess(float[] rawLandmarks, float conf, float handedness,
                                       double[] rotatedPalmBboxFinal, double angleDeg,
                                       AffineUtil.Mat2x3 rotMatrix, double[] padBias) {
        double whX = rotatedPalmBboxFinal[2] - rotatedPalmBboxFinal[0];
        double whY = rotatedPalmBboxFinal[3] - rotatedPalmBboxFinal[1];
        double scaleFactor = Math.max(whX / INPUT_SIZE, whY / INPUT_SIZE);

        AffineUtil.Mat2x3 coordsRot = AffineUtil.rotationMatrix2D(0, 0, angleDeg);
        AffineUtil.Mat2x3 inv = AffineUtil.invert(rotMatrix);

        double centerX = (rotatedPalmBboxFinal[0] + rotatedPalmBboxFinal[2]) / 2;
        double centerY = (rotatedPalmBboxFinal[1] + rotatedPalmBboxFinal[3]) / 2;
        // original_center = inverse of the image-rotation matrix applied to the box center
        double origCenterX = centerX * inv.m00 + centerY * inv.m01 + inv.m02;
        double origCenterY = centerX * inv.m10 + centerY * inv.m11 + inv.m12;

        List<Point3D> points = new ArrayList<>(21);
        for (int i = 0; i < 21; i++) {
            double x = (rawLandmarks[i * 3] - INPUT_SIZE / 2.0) * scaleFactor;
            double y = (rawLandmarks[i * 3 + 1] - INPUT_SIZE / 2.0) * scaleFactor;
            double z = rawLandmarks[i * 3 + 2] * scaleFactor;

            // apply inverse rotation (transpose of the pure-rotation coordsRot) to the offset vector
            double rotX = x * coordsRot.m00 + y * coordsRot.m10;
            double rotY = x * coordsRot.m01 + y * coordsRot.m11;

            double finalX = rotX + origCenterX + padBias[0];
            double finalY = rotY + origCenterY + padBias[1];

            points.add(new Point3D(finalX, finalY, z));
        }

        return new HandLandmarks(points, handedness, conf);
    }

    private static class CropResult {
        BufferedImage image;
        double[] box;   // clipped/enlarged box used for the crop, in the *input* image's coord space
        double[] bias;  // {left, top} offset - subtract from box to get crop-local coords
    }

    private CropResult cropAndPadFromPalm(BufferedImage image, double[] bbox, boolean forRotation) {
        double whX = bbox[2] - bbox[0];
        double whY = bbox[3] - bbox[1];

        double[] shiftVec = forRotation ? PRE_SHIFT : SHIFT;
        double shiftX = shiftVec[0] * whX, shiftY = shiftVec[1] * whY;
        double[] shifted = {bbox[0] + shiftX, bbox[1] + shiftY, bbox[2] + shiftX, bbox[3] + shiftY};

        double centerX = (shifted[0] + shifted[2]) / 2;
        double centerY = (shifted[1] + shifted[3]) / 2;
        whX = shifted[2] - shifted[0];
        whY = shifted[3] - shifted[1];

        double enlarge = forRotation ? PRE_ENLARGE : ENLARGE;
        double halfW = whX * enlarge / 2, halfH = whY * enlarge / 2;

        int x1 = clamp((int) (centerX - halfW), 0, image.getWidth());
        int y1 = clamp((int) (centerY - halfH), 0, image.getHeight());
        int x2 = clamp((int) (centerX + halfW), 0, image.getWidth());
        int y2 = clamp((int) (centerY + halfH), 0, image.getHeight());

        int cropW = Math.max(1, x2 - x1);
        int cropH = Math.max(1, y2 - y1);
        BufferedImage cropped = image.getSubimage(x1, y1, cropW, cropH);

        int sideLen = forRotation
            ? (int) Math.round(Math.hypot(cropH, cropW))
            : Math.max(cropH, cropW);

        int padH = Math.max(0, sideLen - cropH);
        int padW = Math.max(0, sideLen - cropW);
        int left = padW / 2, top = padH / 2;
        int right = padW - left, bottom = padH - top;

        BufferedImage padded = new BufferedImage(cropW + left + right, cropH + top + bottom, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = padded.createGraphics();
        g.fillRect(0, 0, padded.getWidth(), padded.getHeight());
        g.drawImage(cropped, left, top, null);
        g.dispose();

        CropResult result = new CropResult();
        result.image = padded;
        result.box = new double[]{x1, y1, x2, y2};
        result.bias = new double[]{x1 - left, y1 - top};
        return result;
    }

    private double[] subtract(double[] box, double[] bias) {
        return new double[]{box[0] - bias[0], box[1] - bias[1], box[2] - bias[0], box[3] - bias[1]};
    }

    private int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
    private double min(double[] a) { double m = a[0]; for (double v : a) m = Math.min(m, v); return m; }
    private double max(double[] a) { double m = a[0]; for (double v : a) m = Math.max(m, v); return m; }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
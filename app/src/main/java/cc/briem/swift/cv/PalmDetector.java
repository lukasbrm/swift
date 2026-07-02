package cc.briem.swift.cv;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class PalmDetector implements AutoCloseable {

    private static final int INPUT_SIZE = 192;
    private static final String INPUT_NAME = "input_1";
    private static final String BOXES_OUTPUT = "Identity";
    private static final String SCORES_OUTPUT = "Identity_1";
    private static final float SCORE_THRESHOLD = 0.5f;
    private static final float NMS_IOU_THRESHOLD = 0.3f;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final double[][] anchors;

    public PalmDetector(String modelPath, String anchorsCsvPath) throws OrtException, IOException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
        this.anchors = loadAnchors(anchorsCsvPath);
    }

    private double[][] loadAnchors(String path) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try(BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while((line = reader.readLine()) != null) {
                line = line.trim();
                if(line.isEmpty()) continue;
                String[] parts = line.split(",");
                rows.add(new double[]{Double.parseDouble(parts[0]), Double.parseDouble(parts[1])});
            }
        }
        return rows.toArray(new double[0][]);
    }

    public List<Detection> detect(BufferedImage image) throws OrtException {
        int origW = image.getWidth();
        int origH = image.getHeight();

        Preprocessed pre = preprocess(image);

        long[] shape = {1, INPUT_SIZE, INPUT_SIZE, 3};

        try(OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pre.data), shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(INPUT_NAME, inputTensor);
            try(OrtSession.Result result = session.run(inputs)) {
                float[][] boxesRaw = (float[][]) ((float[][][]) result.get(BOXES_OUTPUT).get().getValue())[0];
                float[][] scoresRaw = (float[][]) ((float[][][]) result.get(SCORES_OUTPUT).get().getValue())[0];

                return decode(boxesRaw, scoresRaw, origW, origH, pre.padBiasX, pre.padBiasY);
            }
        }
    }

    private Preprocessed preprocess(BufferedImage image) {
        int w = image.getWidth(), h = image.getHeight();
        double ratio = Math.min(INPUT_SIZE / (double) h, INPUT_SIZE / (double) w);
        int resizedW = (int) Math.round(w * ratio);
        int resizedH = (int) Math.round(h * ratio);
 
        BufferedImage resized = ImageUtil.resize(image, resizedW, resizedH);
 
        int padW = INPUT_SIZE - resizedW;
        int padH = INPUT_SIZE - resizedH;
        int left = padW / 2;
        int top = padH / 2;
 
        BufferedImage canvas = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        g.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE); // black background
        g.drawImage(resized, left, top, null);
        g.dispose();
 
        Preprocessed result = new Preprocessed();
        result.padBiasX = left / ratio;
        result.padBiasY = top / ratio;
 
        float[] data = new float[INPUT_SIZE * INPUT_SIZE * 3];
        int idx = 0;
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                int rgb = canvas.getRGB(x, y);
                data[idx++] = ((rgb >> 16) & 0xFF) / 255f;
                data[idx++] = ((rgb >> 8) & 0xFF) / 255f;
                data[idx++] = (rgb & 0xFF) / 255f;
            }
        }
        result.data = data;
        return result;
    }

    private List<Detection> decode(float[][] boxesRaw, float[][] scoresRaw,
                                    int origW, int origH, double padBiasX, double padBiasY) {
        double scale = Math.max(origW, origH);
        List<Detection> candidates = new ArrayList<>();
 
        for (int i = 0; i < anchors.length - 1; i++) {
            float score = sigmoid(scoresRaw[i][0]);
            if (score < SCORE_THRESHOLD) continue;
 
            float[] box = boxesRaw[i]; // 18 values: [cx,cy,w,h, 7x(lx,ly)]
            double ax = anchors[i][0], ay = anchors[i][1];
 
            double cxDelta = box[0] / INPUT_SIZE;
            double cyDelta = box[1] / INPUT_SIZE;
            double wDelta = box[2] / INPUT_SIZE;
            double hDelta = box[3] / INPUT_SIZE;
 
            double xmin = (cxDelta - wDelta / 2 + ax) * scale - padBiasX;
            double ymin = (cyDelta - hDelta / 2 + ay) * scale - padBiasY;
            double xmax = (cxDelta + wDelta / 2 + ax) * scale - padBiasX;
            double ymax = (cyDelta + hDelta / 2 + ay) * scale - padBiasY;
 
            AffineUtil.Point2D[] landmarks = new AffineUtil.Point2D[7];
            for (int k = 0; k < 7; k++) {
                double lx = (box[4 + k * 2] / INPUT_SIZE + ax) * scale - padBiasX;
                double ly = (box[4 + k * 2 + 1] / INPUT_SIZE + ay) * scale - padBiasY;
                landmarks[k] = new AffineUtil.Point2D(lx, ly);
            }
 
            candidates.add(new Detection(xmin, ymin, xmax, ymax, score, landmarks));
        }
 
        return nonMaxSuppression(candidates);
    }

    private List<Detection> nonMaxSuppression(List<Detection> candidates) {
        candidates.sort((a, b) -> Float.compare(b.score, a.score));
        List<Detection> kept = new ArrayList<>();
        for (Detection c : candidates) {
            boolean overlaps = false;
            for (Detection k : kept) {
                if (iou(c, k) > NMS_IOU_THRESHOLD) { overlaps = true; break; }
            }
            if (!overlaps) kept.add(c);
        }
        return kept;
    }

    private double iou(Detection a, Detection b) {
        double ix1 = Math.max(a.xmin, b.xmin), iy1 = Math.max(a.ymin, b.ymin);
        double ix2 = Math.min(a.xmax, b.xmax), iy2 = Math.min(a.ymax, b.ymax);
        double iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
        double inter = iw * ih;
        double areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        double areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        return inter / (areaA + areaB - inter);
    }

    private float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }

    private static class Preprocessed {
        float[] data;
        double padBiasX, padBiasY;
    }

    public static class Detection {
        public final double xmin, ymin, xmax, ymax;
        public final float score;
        public final AffineUtil.Point2D[] landmarks;

        public Detection(double xmin, double ymin, double xmax, double ymax, float score, AffineUtil.Point2D[] landmarks) {
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
            this.score = score;
            this.landmarks = landmarks;
        }
    }

}
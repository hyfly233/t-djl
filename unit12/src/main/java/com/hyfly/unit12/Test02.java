package com.hyfly.unit12;

import ai.djl.modality.cv.*;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import com.hyfly.utils.ImageUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class Test02 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            Image img = ImageFactory.getInstance()
                    .fromUrl("https://raw.githubusercontent.com/d2l-ai/d2l-en/master/img/catdog.jpg");

            // Width and Height of catdog.jpg
            int WIDTH = img.getWidth();
            int HEIGHT = img.getHeight();
            System.out.println(WIDTH);
            System.out.println(HEIGHT);

            List<Float> sizes = Arrays.asList(0.75f, 0.5f, 0.25f);
            List<Float> ratios = Arrays.asList(1f, 2f, 0.5f);

            MultiBoxPrior mbp = MultiBoxPrior.builder().setSizes(sizes).setRatios(ratios).build();
            NDArray X = manager.randomUniform(0, 1, new Shape(1, 3, HEIGHT, WIDTH));
            NDArray Y = mbp.generateAnchorBoxes(X);
            Y.getShape();

            NDArray boxes = Y.reshape(HEIGHT, WIDTH, 5, 4);
            boxes.get(250, 250, 0);

            Image img2 = img.duplicate();
            drawBBoxes(img2, boxes.get(250, 250),
                    new String[]{"s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2", "s=0.75, r=0.5"});

            img2.getWrappedImage();

            //
            NDArray groundTruth = manager.create(new float[][]{{0, 0.1f, 0.08f, 0.52f, 0.92f},
                    {1, 0.55f, 0.2f, 0.9f, 0.88f}});
            NDArray anchors = manager.create(new float[][]{{0, 0.1f, 0.2f, 0.3f}, {0.15f, 0.2f, 0.4f, 0.4f},
                    {0.63f, 0.05f, 0.88f, 0.98f}, {0.66f, 0.45f, 0.8f, 0.8f},
                    {0.57f, 0.3f, 0.92f, 0.9f}});

            Image img3 = img.duplicate();

            drawBBoxes(img3, groundTruth.get(new NDIndex(":, 1:")), new String[]{"dog", "cat"});
            drawBBoxes(img3, anchors, new String[]{"0", "1", "2", "3", "4"});

            img3.getWrappedImage();

            //
            MultiBoxTarget mbt = MultiBoxTarget.builder().build();
            NDList labels = mbt.target(new NDList(anchors.expandDims(0),
                    groundTruth.expandDims(0),
                    manager.zeros(new Shape(1, 3, 5))));

            NDArray ndArray = labels.get(2);
            log.info(ndArray.toDebugString(true));

            ndArray = labels.get(0);
            log.info(ndArray.toDebugString(true));

            //
            anchors = manager.create(new float[][]{{0.1f, 0.08f, 0.52f, 0.92f}, {0.08f, 0.2f, 0.56f, 0.95f},
                    {0.15f, 0.3f, 0.62f, 0.91f}, {0.55f, 0.2f, 0.9f, 0.88f}});
            NDArray offsetPreds = manager.zeros(new Shape(16));
            NDArray clsProbs = manager.create(new float[][]{{0, 0, 0, 0}, // Predicted Probability for Background
                    {0.9f, 0.8f, 0.7f, 0.1f}, // Predicted Probability for Dog
                    {0.1f, 0.2f, 0.3f, 0.9f}}); // Predicted Probability for Cat

            Image img4 = img.duplicate();
            drawBBoxes(img4, anchors, new String[]{"dog=0.9", "dog=0.8", "dog=0.7", "cat=0.9"});

            img4.getWrappedImage();

            MultiBoxDetection mbd = MultiBoxDetection.builder().optThreshold(0.5f).build();
            NDArray output = mbd.detection(new NDList(clsProbs.expandDims(0),
                    offsetPreds.expandDims(0),
                    anchors.expandDims(0))).head(); // shape = 1, 4, 6
            log.info(output.toDebugString(true));

            //
            Image img5 = img.duplicate();
            for (int i = 0; i < output.size(1); i++) {
                NDArray bbox = output.get(0, i);
                // Skip prediction bounding boxes of category -1
                if (bbox.getFloat(0) == -1) {
                    continue;
                }
                String[] labels1 = {"dog=", "cat="};
                String className = labels1[(int) bbox.getFloat(0)];
                String prob = Float.toString(bbox.getFloat(1));
                String label = className + prob;
                drawBBoxes(img5, bbox.reshape(1, bbox.size()).get(new NDIndex(":, 2:")), new String[]{label});
            }

            img5.getWrappedImage();

        }
    }

    /*
     * Draw Bounding Boxes on Image
     *
     * Saved in ImageUtils
     */
    public static void drawBBoxes(Image img, NDArray boxes, String[] labels) {
        if (labels == null) {
            labels = new String[(int) boxes.size(0)];
            Arrays.fill(labels, "");
        }

        List<String> classNames = new ArrayList();
        List<Double> prob = new ArrayList();
        List<BoundingBox> boundBoxes = new ArrayList();
        for (int i = 0; i < boxes.size(0); i++) {
            NDArray box = boxes.get(i);
            Rectangle rect = ImageUtils.bboxToRect(box);
            classNames.add(labels[i]);
            prob.add(1.0);
            boundBoxes.add(rect);
        }
        DetectedObjects detectedObjects = new DetectedObjects(classNames, prob, boundBoxes);
        img.drawBoundingBoxes(detectedObjects);
    }
}

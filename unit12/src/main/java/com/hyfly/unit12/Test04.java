package com.hyfly.unit12;

import ai.djl.basicdataset.cv.BananaDetection;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import lombok.extern.slf4j.Slf4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;


@Slf4j
public class Test04 {


    public static void main(String[] args) throws Exception {
        // Load the bananas dataset.
        BananaDetection trainIter = BananaDetection.builder()
                .setSampling(32, true)  // Read the dataset in random order
                .optUsage(Dataset.Usage.TRAIN)
                .build();

        trainIter.prepare();

        try (NDManager manager = NDManager.newBaseManager()) {
            Batch batch = trainIter.getData(manager).iterator().next();
            System.out.println(batch.getData().get(0).getShape() + ", " + batch.getLabels().get(0).getShape());

//            Image[] imageArr = new Image[10];
//            List<List<String>> classNames = new ArrayList<>();
//            List<List<Double>> prob = new ArrayList<>();
//            List<List<BoundingBox>> boxes = new ArrayList<>();
//
//            batch = trainIter.getData(manager).iterator().next();
//            for (int i = 0; i < 10; i++) {
//                NDArray imgData = batch.getData().get(0).get(i);
//                imgData.muli(255);
//                NDArray imgLabel = batch.getLabels().get(0).get(i);
//
//                List<String> bananaList = new ArrayList<>();
//                bananaList.add("banana");
//                classNames.add(new ArrayList<>(bananaList));
//
//                List<Double> probabilityList = new ArrayList<>();
//                probabilityList.add(1.0);
//                prob.add(new ArrayList<>(probabilityList));
//
//                List<BoundingBox> boundBoxes = new ArrayList<>();
//
//                float[] coord = imgLabel.get(0).toFloatArray();
//                double first = (double) (coord[1]);
//                double second = (double) (coord[2]);
//                double third = (double) (coord[3]);
//                double fourth = (double) (coord[4]);
//
//                boundBoxes.add(new Rectangle(first, second, (third - first), (fourth - second)));
//
//                boxes.add(new ArrayList<>(boundBoxes));
//                DetectedObjects detectedObjects = new DetectedObjects(classNames.get(i), prob.get(i), boxes.get(i));
//                imageArr[i] = ImageFactory.getInstance().fromNDArray(imgData.toType(DataType.INT8, true));
//                imageArr[i].drawBoundingBoxes(detectedObjects);
//            }
//
//            // refer to https://github.com/deepjavalibrary/d2l-java/tree/master/documentation/troubleshoot.md
//            // if you encounter X11 errors when drawing bounding boxes.
//            System.out.println(showImages(imageArr, 256, 256));
        }
    }

    public static BufferedImage showImages(Image[] dataset, int width, int height) {
        int col = 1280 / width;
        int row = (dataset.length + col - 1) / col;
        int w = col * (width + 3);
        int h = row * (height + 3);
        BufferedImage bi = new BufferedImage(w + 3, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = bi.createGraphics();

        for (int i = 0; i < dataset.length; i++) {
            Image image = dataset[i];
            BufferedImage img = (BufferedImage) image.getWrappedImage();
            int x = (i % col) * (width + 3) + 3;
            int y = (i / col) * (height + 3) + 3;
            g.drawImage(img, x, y, width, height, null);
        }
        g.dispose();
        return bi;
    }
}

package com.hyfly.unit02;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import cn.hutool.core.date.StopWatch;
import com.hyfly.utils.ImageUtils;
import lombok.extern.slf4j.Slf4j;

import java.awt.*;
import java.awt.image.BufferedImage;

@Slf4j
public class Test06 {

    public static void main(String[] args) throws Exception {
        int batchSize = 256;
        boolean randomShuffle = true;

        FashionMnist mnistTrain = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        FashionMnist mnistTest = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        mnistTrain.prepare();
        mnistTest.prepare();

        log.info(String.valueOf(mnistTrain.size()));
        log.info(String.valueOf(mnistTest.size()));

        try (NDManager manager = NDManager.newBaseManager()) {
            final int SCALE = 4;
            final int WIDTH = 28;
            final int HEIGHT = 28;

            showImages(mnistTrain, 6, WIDTH, HEIGHT, SCALE, manager);

            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            for (Batch batch : mnistTrain.getData(manager)) {
                NDArray x = batch.getData().head();
                NDArray y = batch.getLabels().head();
            }
            stopWatch.stop();
            System.out.printf("%.2f sec%n", stopWatch.getTotalTimeSeconds());
        }

    }

    // Saved in the FashionMnist class for later use
    public static String[] getFashionMnistLabels(int[] labelIndices) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        String[] convertedLabels = new String[labelIndices.length];
        for (int i = 0; i < labelIndices.length; i++) {
            convertedLabels[i] = textLabels[labelIndices[i]];
        }
        return convertedLabels;
    }

    public static String getFashionMnistLabel(int labelIndice) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        return textLabels[labelIndice];
    }

    // Saved in the FashionMnistUtils class for later use
    public static BufferedImage showImages(
            ArrayDataset dataset, int number, int width, int height, int scale, NDManager manager) {
        BufferedImage[] images = new BufferedImage[number];
        String[] labels = new String[number];
        for (int i = 0; i < number; i++) {
            Record record = dataset.get(manager, i);
            NDArray array = record.getData().get(0).squeeze(-1);
            int y = (int) record.getLabels().get(0).getFloat();
            images[i] = toImage(array, width, height);
            labels[i] = getFashionMnistLabel(y);
        }
        int w = images[0].getWidth() * scale;
        int h = images[0].getHeight() * scale;

        return ImageUtils.showImages(images, labels, w, h);
    }

    private static BufferedImage toImage(NDArray array, int width, int height) {
        System.setProperty("apple.awt.UIElement", "true");
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = (Graphics2D) img.getGraphics();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float c = array.getFloat(j, i) / 255; // scale down to between 0 and 1
                g.setColor(new Color(c, c, c)); // set as a gray color
                g.fillRect(i, j, 1, 1);
            }
        }
        g.dispose();
        return img;
    }
}

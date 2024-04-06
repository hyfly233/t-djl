package com.hyfly.unit02;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import com.hyfly.utils.FashionMnistUtils;
import com.hyfly.utils.StopWatch;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test06 {

    public static void main(String[] args) throws Exception {
        // 3.5.1. 读取数据集
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

            FashionMnistUtils.showImages(mnistTrain, 6, WIDTH, HEIGHT, SCALE, manager);

            // 3.5.2. 读取小批量
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            for (Batch batch : mnistTrain.getData(manager)) {
                try (NDList dataX = batch.getData();
                     NDList dataY = batch.getLabels()) {
                    NDArray x = dataX.head();
                    NDArray y = dataY.head();
                    log.info("x: {}", x.getShape());
                    log.info("y: {}", y.getShape());
                }
            }
            log.info("{} sec", stopWatch.stop());
        }

    }
}

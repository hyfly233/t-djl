package com.hyfly.unit02;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.hyfly.utils.DataPoints;
import com.hyfly.utils.Training;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 3.3. 线性回归的简洁实现
 */
@Slf4j
public class Test05 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            log.info("3.3.1. 生成数据 ------------");
            NDArray trueW = manager.create(new float[]{2, -3.4f});
            log.info("trueW: {}", trueW.toDebugString(true));

            float trueB = 4.2f;

            DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB, 1000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();

            log.info("3.3.2. 读取数据 ------------");
            int batchSize = 10;
            ArrayDataset dataset = Training.loadArray(features, labels, batchSize, false);

            Batch batch = dataset.getData(manager).iterator().next();
            NDArray x = batch.getData().head();
            NDArray y = batch.getLabels().head();
            log.info(x.toDebugString(true));
            log.info(y.toDebugString(true));
            batch.close();

            log.info("3.3.3. 定义模型 ------------");
            Model model = Model.newInstance("lin-reg");

            SequentialBlock net = new SequentialBlock();
            Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
            net.add(linearBlock);

            model.setBlock(net);

            log.info("3.3.4. 定义损失函数 平方损失 ------------");
            Loss l2loss = Loss.l2Loss();

            log.info("3.3.5. 定义优化算法 小批量随机梯度下降 ------------");
            Tracker lrt = Tracker.fixed(0.03f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            log.info("3.3.6. Trainer 的初始化配置 ------------");
            DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
                    .optOptimizer(sgd) // Optimizer (loss function)
                    .optDevices(manager.getEngine().getDevices(1)) // single GPU
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            log.info("3.3.7. 初始化模型参数 ------------");
            // First axis is batch size - won't impact parameter initialization
            // Second axis is the input size
            trainer.initialize(new Shape(batchSize, 2));

            log.info("3.3.8. 运行性能指标 ------------");
            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);

            log.info("3.3.9. 训练 ------------");
            int numEpochs = 3;

            for (int epoch = 1; epoch <= numEpochs; epoch++) {
                log.info("Epoch {}", epoch);
                // Iterate over dataset
                for (Batch batch01 : trainer.iterateDataset(dataset)) {
                    // Update loss and evaulator
                    EasyTrain.trainBatch(trainer, batch01);

                    // Update parameters
                    trainer.step();

                    batch01.close();
                }
                // reset training and validation evaluators at end of epoch
                trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            }

            Block layer = model.getBlock();
            ParameterList params = layer.getParameters();
            try (Parameter parameter0 = params.valueAt(0);
                 Parameter parameter1 = params.valueAt(1)) {
                NDArray wParam = parameter0.getArray();
                NDArray bParam = parameter1.getArray();

                float[] w = trueW.sub(wParam.reshape(trueW.getShape())).toFloatArray();
                log.info("Error in estimating w: [{} {}]", w[0], w[1]);
                log.info("Error in estimating b: {}", trueB - bParam.getFloat());

                log.info("3.3.10. 保存训练模型 ------------");
                Path modelDir = Paths.get("./models/lin-reg");
                Files.createDirectories(modelDir);

                model.setProperty("Epoch", Integer.toString(numEpochs)); // save epochs trained as metadata
                model.save(modelDir, "lin-reg");

                log.info(model.toString());
            }
        }
    }
}

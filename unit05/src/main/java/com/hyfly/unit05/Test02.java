package com.hyfly.unit05;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test02 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()){
            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 8, 8));

            // 请注意，这里每侧填充1行或1列，因此总共2行或1列
            // 添加行或列
            Block block = Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .optPadding(new Shape(1, 1))
                    .setFilters(1)
                    .build();

            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss());
            Model model = Model.newInstance("conv2D");
            model.setBlock(block);

            Trainer trainer = model.newTrainer(config);
            trainer.initialize(X.getShape());

            NDArray yHat = trainer.forward(new NDList(X)).singletonOrThrow();
            // 排除我们不感兴趣的前两个维度：批次和
            // 频道
            System.out.println(yHat.getShape().slice(2));

            // 这里，我们使用一个高度为5、宽度为3的卷积核。这个
            // 高度和宽度两侧的填充号分别为2和1，
            // 分别

            block = Conv2d.builder()
                    .setKernelShape(new Shape(5, 3))
                    .optPadding(new Shape(2, 1))
                    .setFilters(1)
                    .build();

            model.setBlock(block);

            trainer = model.newTrainer(config);
            trainer.initialize(X.getShape());

            yHat = trainer.forward(new NDList(X)).singletonOrThrow();
            System.out.println(yHat.getShape().slice(2));

            //
            block = Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .optPadding(new Shape(1, 1))
                    .optStride(new Shape(2,2))
                    .setFilters(1)
                    .build();

            model.setBlock(block);

            trainer = model.newTrainer(config);
            trainer.initialize(X.getShape());

            yHat = trainer.forward(new NDList(X)).singletonOrThrow();
            System.out.println(yHat.getShape().slice(2));

            //
            block = Conv2d.builder()
                    .setKernelShape(new Shape(3, 5))
                    .optPadding(new Shape(0, 1))
                    .optStride(new Shape(3,4))
                    .setFilters(1)
                    .build();

            model.setBlock(block);

            trainer = model.newTrainer(config);
            trainer.initialize(X.getShape());

            yHat = trainer.forward(new NDList(X)).singletonOrThrow();
            System.out.println(yHat.getShape().slice(2));
        }
    }
}

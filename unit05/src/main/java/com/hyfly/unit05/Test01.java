package com.hyfly.unit05;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.loss.Loss;
import com.hyfly.utils.Corr2d;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test01 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.create(new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8}, new Shape(3, 3));
            NDArray K = manager.create(new float[]{0, 1, 2, 3}, new Shape(2, 2));
            System.out.println(Corr2d.corr2d(manager, X, K));

            //
            X = manager.ones(new Shape(6, 8));
            X.set(new NDIndex(":" + "," + 2 + ":" + 6), 0f);
            System.out.println(X);

            //
            K = manager.create(new float[]{1, -1}, new Shape(1, 2));

            //
            NDArray Y = Corr2d.corr2d(manager, X, K);
            log.info(Y.toDebugString(true));

            //
            X = X.reshape(1, 1, 6, 8);
            Y = Y.reshape(1, 1, 6, 7);

            Loss l2Loss = Loss.l2Loss();

            // 构造一个具有1个输出通道和一个
            // 形核（1，2）。为了简单起见，我们忽略了这里的偏见
            Block block = Conv2d.builder()
                    .setKernelShape(new Shape(1, 2))
                    .optBias(false)
                    .setFilters(1)
                    .build();

            block.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
            block.initialize(manager, DataType.FLOAT32, X.getShape());

            // 二维卷积层使用四维输入和输出
            // 输出格式为（例如，通道、高度、宽度），其中批次
            // 大小（批次中的示例数）和通道数均为1

            ParameterList params = block.getParameters();
            NDArray wParam = params.get(0).getValue().getArray();
            wParam.setRequiresGradient(true);

            NDArray lossVal = null;
            ParameterStore parameterStore = new ParameterStore(manager, false);


            for (int i = 0; i < 10; i++) {

                wParam.setRequiresGradient(true);

                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    NDArray yHat = block.forward(parameterStore, new NDList(X), true).singletonOrThrow();
                    NDArray l = l2Loss.evaluate(new NDList(Y), new NDList(yHat));
                    lossVal = l;
                    gc.backward(l);
                }
                // 更新内核
                wParam.subi(wParam.getGradient().mul(0.40f));

                if ((i + 1) % 2 == 0) {
                    System.out.println("batch " + (i + 1) + " loss: " + lossVal.sum().getFloat());
                }
            }

            ParameterList params1 = block.getParameters();
            NDArray wParam1 = params1.get(0).getValue().getArray();
            log.info("wParam1: {}", wParam1.toDebugString(true));
        }
    }
}

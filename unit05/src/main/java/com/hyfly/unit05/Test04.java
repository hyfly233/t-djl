package com.hyfly.unit05;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test04 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.arange(9f).reshape(3, 3);
            NDArray ndArray = pool2d(manager, X, new Shape(2, 2), "max");

            log.info("{}", ndArray.toDebugString(true));

            ndArray = pool2d(manager, X, new Shape(2, 2), "avg");
            log.info("{}", ndArray.toDebugString(true));

            X = manager.arange(16f).reshape(1, 1, 4, 4);
            log.info("{}", X.toDebugString(true));

            // 定义块指定内核和步幅
            Block block = Pool.maxPool2dBlock(new Shape(3, 3), new Shape(3, 3));
            block.initialize(manager, DataType.FLOAT32, new Shape(1, 1, 4, 4));

            ParameterStore parameterStore = new ParameterStore(manager, false);
            // 因为池层中没有模型参数，所以我们不需要
            // 调用参数初始化函数
            ndArray = block.forward(parameterStore, new NDList(X), true).singletonOrThrow();
            log.info("{}", ndArray.toDebugString(true));

            // 重新定义内核形状、跨步形状和垫块形状
            block = Pool.maxPool2dBlock(new Shape(2, 3), new Shape(2, 3), new Shape(1, 2));
            ndArray = block.forward(parameterStore, new NDList(X), true).singletonOrThrow();
            log.info("{}", ndArray.toDebugString(true));

            X = X.concat(X.add(1), 1);
            log.info("{}", X.toDebugString(true));

            block = Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1));
            ndArray = block.forward(parameterStore, new NDList(X), true).singletonOrThrow();
            log.info("{}", ndArray.toDebugString(true));
        }
    }

    public static NDArray pool2d(NDManager manager, NDArray X, Shape poolShape, String mode) {

        long poolHeight = poolShape.get(0);
        long poolWidth = poolShape.get(1);

        NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - poolHeight + 1,
                X.getShape().get(1) - poolWidth + 1));
        for (int i = 0; i < Y.getShape().get(0); i++) {
            for (int j = 0; j < Y.getShape().get(1); j++) {

                if ("max".equals(mode)) {
                    Y.set(new NDIndex(i + "," + j),
                            X.get(new NDIndex(i + ":" + (i + poolHeight) + ", " + j + ":" + (j + poolWidth))).max());
                } else if ("avg".equals(mode)) {
                    Y.set(new NDIndex(i + "," + j),
                            X.get(new NDIndex(i + ":" + (i + poolHeight) + ", " + j + ":" + (j + poolWidth))).mean());
                }

            }
        }

        return Y;
    }
}

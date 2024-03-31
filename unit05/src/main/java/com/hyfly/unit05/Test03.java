package com.hyfly.unit05;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test03 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {

            NDArray X = manager.create(new Shape(2, 3, 3), DataType.INT32);
            X.set(new NDIndex(0), manager.arange(9));
            X.set(new NDIndex(1), manager.arange(1, 10));
            X = X.toType(DataType.FLOAT32, true);

            NDArray K = manager.create(new Shape(2, 2, 2), DataType.INT32);
            K.set(new NDIndex(0), manager.arange(4));
            K.set(new NDIndex(1), manager.arange(1, 5));
            K = K.toType(DataType.FLOAT32, true);

            NDArray ndArray = corr2dMultiIn(manager, X, K);
            log.info(ndArray.toDebugString(true));

            //
            K = NDArrays.stack(new NDList(K, K.add(1), K.add(2)));
            K.getShape();

            ndArray = corrMultiInOut(manager, X, K);
            log.info(ndArray.toDebugString(true));

            //
            X = manager.randomUniform(0f, 1.0f, new Shape(3, 3, 3));
            K = manager.randomUniform(0f, 1.0f, new Shape(2, 3, 1, 1));

            NDArray Y1 = corr2dMultiInOut1x1(X, K);
            NDArray Y2 = corrMultiInOut(manager, X, K);

            System.out.println(Math.abs(Y1.sum().getFloat() - Y2.sum().getFloat()) < 1e-6);
        }
    }

    public static NDArray corr2D(NDManager manager, NDArray X, NDArray K) {

        long h = K.getShape().get(0);
        long w = K.getShape().get(1);

        NDArray Y = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));

        for (int i = 0; i < Y.getShape().get(0); i++) {
            for (int j = 0; j < Y.getShape().get(1); j++) {
                NDArray temp = X.get(i + ":" + (i + h) + "," + j + ":" + (j + w)).mul(K);
                Y.set(new NDIndex(i + "," + j), temp.sum());
            }
        }
        return Y;
    }

    public static NDArray corr2dMultiIn(NDManager manager, NDArray X, NDArray K) {

        long h = K.getShape().get(0);
        long w = K.getShape().get(1);

        // 首先，沿着'X'的第0维（通道维）进行遍历
        // “K”。然后，把它们加在一起

        NDArray res = manager.zeros(new Shape(X.getShape().get(0) - h + 1, X.getShape().get(1) - w + 1));
        for (int i = 0; i < X.getShape().get(0); i++) {
            for (int j = 0; j < K.getShape().get(0); j++) {
                if (i == j) {
                    res = res.add(corr2D(manager, X.get(new NDIndex(i)), K.get(new NDIndex(j))));
                }
            }
        }
        return res;
    }

    public static NDArray corrMultiInOut(NDManager manager, NDArray X, NDArray K) {

        long cin = K.getShape().get(0);
        long h = K.getShape().get(2);
        long w = K.getShape().get(3);

        // 沿“K”的第0维遍历，每次执行
        // 输入'X'的互相关运算。所有结果都是正确的
        // 使用stack函数合并在一起

        NDArray res = manager.create(new Shape(cin, X.getShape().get(1) - h + 1, X.getShape().get(2) - w + 1));

        for (int j = 0; j < K.getShape().get(0); j++) {
            res.set(new NDIndex(j), corr2dMultiIn(X, K.get(new NDIndex(j))));
        }

        return res;
    }

    public static NDArray corr2dMultiInOut1x1(NDArray X, NDArray K) {

        long channelIn = X.getShape().get(0);
        long height = X.getShape().get(1);
        long width = X.getShape().get(2);

        long channelOut = K.getShape().get(0);
        X = X.reshape(channelIn, height * width);
        K = K.reshape(channelOut, channelIn);
        NDArray Y = K.dot(X); // 全连通层中的矩阵乘法

        return Y.reshape(channelOut, height, width);
    }
}

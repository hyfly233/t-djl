package com.hyfly.unit01;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test05 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(4f);
            System.out.println(x.toDebugString(true));

            // 为 NDArray 的梯度分配内存
            x.setRequiresGradient(true);
            x.getGradient();

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.dot(x).mul(2);
                log.info(y.toDebugString(true));
                gc.backward(y);
            }
            log.info(x.getGradient().toDebugString(true));

            log.info(x.getGradient().eq(x.mul(4)).toDebugString(true));

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.sum();
                gc.backward(y);
            }
            log.info(x.getGradient().toDebugString(true));

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.mul(x); // y 是一个向量
                gc.backward(y);
            }
            log.info(x.getGradient().toDebugString(true)); // 等价于y = sum(x * x)

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.mul(x);
                NDArray u = y.stopGradient();
                NDArray z = u.mul(x);
                gc.backward(z);
                System.out.println(x.getGradient().eq(u).toDebugString(true));
            }

            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray y = x.mul(x);
                y = x.mul(x);
                gc.backward(y);
                System.out.println(x.getGradient().eq(x.mul(2)).toDebugString(true));
            }

            NDArray a = manager.randomNormal(new Shape(1));
            a.setRequiresGradient(true);
            try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                NDArray d = f(a);
                gc.backward(d);

                System.out.println(a.getGradient().eq(d.div(a)).toDebugString(true));
            }
        }
    }

    public static NDArray f(NDArray a) {
        NDArray b = a.mul(2);
        while (b.norm().getFloat() < 1000) {
            b = b.mul(2);
        }
        NDArray c;
        if (b.sum().getFloat() > 0) {
            c = b;
        } else {
            c = b.mul(100);
        }
        return c;
    }
}

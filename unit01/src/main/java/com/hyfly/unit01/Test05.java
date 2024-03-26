package com.hyfly.unit01;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test05 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(4f);

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
        }
    }
}

package com.hyfly.unit10.entity;

import ai.djl.ndarray.NDList;

import java.util.Map;

@FunctionalInterface
public interface TrainerConsumer {
    void train(NDList params, NDList states, Map<String, Float> hyperparams);

}

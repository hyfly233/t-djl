package com.hyfly.unit10.entity;

public class SquareRootTracker {
    float lr;

    public SquareRootTracker() {
        this(0.1f);
    }

    public SquareRootTracker(float learningRate) {
        this.lr = learningRate;
    }

    public float getNewLearningRate(int numUpdate) {
        return lr * (float) Math.pow(numUpdate + 1, -0.5);
    }
}

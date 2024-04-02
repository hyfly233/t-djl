package com.hyfly.unit10.entity;

public class DemoFactorTracker {
    float baseLr;
    float stopFactorLr;
    float factor;

    public DemoFactorTracker(float factor, float stopFactorLr, float baseLr) {
        this.factor = factor;
        this.stopFactorLr = stopFactorLr;
        this.baseLr = baseLr;
    }

    public DemoFactorTracker() {
        this(1f, (float) 1e-7, 0.1f);
    }

    public float getNewLearningRate(float lr, int numUpdate) {
        return lr * (float) Math.pow(numUpdate + 1, -0.5);
    }
}

package com.hyfly.unit10.entity;

public class CosineWarmupTracker {
    float baseLr;
    float finalLr;
    int maxUpdate;
    int warmUpSteps;
    float warmUpBeginValue;
    float warmUpFinalValue;

    public CosineWarmupTracker() {
        this(0.5f, 0.01f, 20, 5);
    }

    public CosineWarmupTracker(float baseLr, float finalLr, int maxUpdate, int warmUpSteps) {
        this.baseLr = baseLr;
        this.finalLr = finalLr;
        this.maxUpdate = maxUpdate;
        this.warmUpSteps = 5;
        this.warmUpBeginValue = 0f;
    }

    public float getNewLearningRate(int numUpdate) {
        if (numUpdate <= warmUpSteps) {
            return getWarmUpValue(numUpdate);
        }
        if (numUpdate > maxUpdate) {
            return finalLr;
        }
        // Scale the cosine curve to fit smoothly with the warmup steps
        float step = (baseLr - finalLr) / 2 * (1 +
            (float) Math.cos(Math.PI * (numUpdate - warmUpSteps) / (maxUpdate - warmUpSteps)));
        return finalLr + step;
    }

    public float getWarmUpValue(int numUpdate) {
        // Linear warmup
        return warmUpBeginValue + (baseLr - warmUpBeginValue) * numUpdate / warmUpSteps;
    }
}

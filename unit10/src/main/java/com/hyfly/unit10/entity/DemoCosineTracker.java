package com.hyfly.unit10.entity;

public class DemoCosineTracker {
    float baseLr;
    float finalLr;
    int maxUpdate;
    public DemoCosineTracker() {
        this(0.5f, 0.01f, 20);
    }
    public DemoCosineTracker(float baseLr, float finalLr, int maxUpdate) {
        this.baseLr = baseLr;
        this.finalLr = finalLr;
        this.maxUpdate = maxUpdate;
    }
    public float getNewLearningRate(int numUpdate) {
        if (numUpdate > maxUpdate) {
            return finalLr;
        }
        // Scale the curve to smoothly transition
        float step = (baseLr - finalLr) / 2 * (1 + (float) Math.cos(Math.PI * numUpdate / maxUpdate));
        return finalLr + step;
    }
}

package com.hyfly.unit10.entity;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

import java.util.Map;

public class Optimization {
    public static void sgd(NDList params, NDList states, Map<String, Float> hyperparams) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            // Update param
            // param = param - param.gradient * lr
            param.subi(param.getGradient().mul(hyperparams.get("lr")));
        }
    }

    public static void sgdMomentum(NDList params, NDList states, Map<String, Float> hyperparams) {
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            NDArray velocity = states.get(i);
            // Update param
            velocity.muli(hyperparams.get("momentum")).addi(param.getGradient());
            param.subi(velocity.mul(hyperparams.get("lr")));
        }
    }

    public static void adagrad(NDList params, NDList states, Map<String, Float> hyperparams) {
        float eps = (float) 1e-6;
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            NDArray state = states.get(i);
            // Update param
            state.addi(param.getGradient().square());
            param.subi(param.getGradient().mul(hyperparams.get("lr")).div(state.add(eps).sqrt()));
        }
    }

    public static void rmsProp(NDList params, NDList states, Map<String, Float> hyperparams) {
        float gamma = hyperparams.get("gamma");
        float eps = (float) 1e-6;
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            NDArray state = states.get(i);
            // Update parameter and state
            // state = gamma * state + (1 - gamma) * param.gradient^(1/2)
            state.muli(gamma).addi(param.getGradient().square().mul(1 - gamma));
            // param -= lr * param.gradient / sqrt(s + eps)
            param.subi(param.getGradient().mul(hyperparams.get("lr")).div(state.add(eps).sqrt()));
        }
    }

    public static void adadelta(NDList params, NDList states, Map<String, Float> hyperparams) {
        float rho = hyperparams.get("rho");
        float eps = (float) 1e-5;
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            NDArray state = states.get(2 * i);
            NDArray delta = states.get(2 * i + 1);
            // Update parameter, state, and delta
            // In-place updates with the '__'i methods (ex. muli)
            // state = rho * state + (1 - rho) * param.gradient^2
            state.muli(rho).addi(param.getGradient().square().mul(1 - rho));
            // rescaledGradient = ((delta + eps)^(1/2) / (state + eps)^(1/2)) * param.gradient
            NDArray rescaledGradient = delta.add(eps).sqrt()
                    .div(state.add(eps).sqrt()).mul(param.getGradient());
            // param -= rescaledGradient
            param.subi(rescaledGradient);
            // delta = rho * delta + (1 - rho) * g^2
            delta.muli(rho).addi(rescaledGradient.square().mul(1 - rho));
        }
    }

    public static void adam(NDList params, NDList states, Map<String, Float> hyperparams) {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = (float) 1e-6;
        float time = hyperparams.get("time");
        for (int i = 0; i < params.size(); i++) {
            NDArray param = params.get(i);
            NDArray velocity = states.get(2 * i);
            NDArray state = states.get(2 * i + 1);
            // Update parameter, velocity, and state
            // velocity = beta1 * v + (1 - beta1) * param.gradient
            velocity.muli(beta1).addi(param.getGradient().mul(1 - beta1));
            // state = beta2 * state + (1 - beta2) * param.gradient^2
            state.muli(beta2).addi(param.getGradient().square().mul(1 - beta2));
            // vBiasCorr = velocity / ((1 - beta1)^(time))
            NDArray vBiasCorr = velocity.div(1 - Math.pow(beta1, time));
            // sBiasCorr = state / ((1 - beta2)^(time))
            NDArray sBiasCorr = state.div(1 - Math.pow(beta2, time));
            // param -= lr * vBiasCorr / (sBiasCorr^(1/2) + eps)
            param.subi(vBiasCorr.mul(hyperparams.get("lr")).div(sBiasCorr.sqrt().add(eps)));
        }
        hyperparams.put("time", time + 1);
    }
}

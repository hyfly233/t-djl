package com.hyfly.unit03;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.util.RandomUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.Random;

@Slf4j
public class Test04 {

    public static void main(String[] args) {

    }

    public static void swap(NDArray arr, int i, int j) {
        float tmp = arr.getFloat(i);
        arr.set(new NDIndex(i), arr.getFloat(j));
        arr.set(new NDIndex(j), tmp);
    }

    public static NDArray shuffle(NDArray arr) {
        int size = (int) arr.size();

        Random rnd = RandomUtils.RANDOM;

        for (int i = Math.toIntExact(size) - 1; i > 0; --i) {
            swap(arr, i, rnd.nextInt(i));
        }
        return arr;
    }
}

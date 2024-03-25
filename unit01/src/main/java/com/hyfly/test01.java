package com.hyfly;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.util.Arrays;

public class test01 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.arange(12);
            System.out.println(Arrays.toString(x.toArray()));
        }
    }
}
